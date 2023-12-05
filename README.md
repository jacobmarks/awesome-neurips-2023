# Awesome NeurIPS 2023 Info

![Neurips 2023 wordcloud](images/wordcloud_2023.png)
Caption: Wordcloud of all NeurIPS 2023 titles

Welcome to the hub for all things [NeurIPS 2023](https://neurips.cc/)! We scraped the data for all 3500+ NeurIPS projects and dove into the depths of Hugging Face, GitHub, LinkedIn, and Arxiv to pick out the most interesting content.

In this repo, you will find:

- [Data Analysis](#data-analysis): detailed analysis of the titles and abstracts from NeurIPS 2023 accepted papers
- [Awesome Projects](#cool-neurips-projects): synthesized collection of 40 NeurIPS 2023 papers you won't want to miss
- [Conference Schedule](#conference-schedule): comprehensive listing of all NeurIPS 2023 projects (title, authors, abstract) organized by poster session and sorted alphabetically

_Note_: Our data contains 3581 out of the 3584 accepted papers. If you are an author of one of the 3 remaining papers, I sincerely apologize, and would love for you to get in touch so we can add the paper :)

## Data Analysis

The raw data is included in this repo. If you have ideas for other interesting analyses, feel free to create an issue or submit a PR!

For now, insights are organized into the following categories:

- Authors
- Titles
- Abstracts

For the data analysis itself, check out the [Jupyter Notebook](./analysis.ipynb)!

<details><summary><h3 style='display: inline;'> Authors</h3></summary>

![Neurips num authors](images/num_authors_2022_2023.png)

#### Most prolific authors

The top 10 authors with the most papers at NeurIPS 2023 are:

- Bo Li: 15 papers
- Ludwig Schmidt: 14 papers
- Mihaela van der Schaar: 13 papers
- Bo Han: 13 papers
- Hao Wang: 12 papers
- Masashi Sugiyama: 11 papers
- Bernhard Schölkopf: 11 papers
- Dacheng Tao: 11 papers
- Andreas Krause: 11 papers
- Tongliang Liu: 11 papers

#### Number of unique authors

There were 12,994 unique authors at NeurIPS 2023, up from 9913 at NeurIPS 2022.

#### Number of authors per paper

- The average number of authors per paper was **4.98**, up from 4.66 at NeurIPS 2022.
- Additionally, there were a handful of single-author papers, in contrast to NeurIPS 2022, where the minimum number of authors was 2.
- The paper with the most authors was [ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation](https://arxiv.org/abs/2306.08754)

</details>

<details><summary><h3 style='display: inline;'> Titles</h3></summary>

#### Title Length

![Neurips 2023 title length histogram](images/title_length_histogram_2022_2023.png)

- The average title length was **8.72** words, up from 8.48 at NeurIPS 2022.

#### Prevalence of Acronyms

22% of titles introduced an acronym, up from 18% at NeurIPS 2022.

#### LaTeX in Titles

- 1.3% of titles contained LaTeX, whereas none of the titles at NeurIPS 2022 contained LaTeX.

</details>

<details><summary><h3 style='display: inline;'> Abstracts</h3></summary>

![abstract length](images/abstract_histogram_2023.png)

#### Abstract Length

- The longest abstract was from [[Re] On the Reproducibility of FairCal: Fairness Calibration for Face Verification](https://neurips.cc/virtual/2023/poster/74168), which has 373 words.
- The shortest abstract was from [Improved Convergence in High Probability of Clipped Gradient Methods with Heavy Tailed Noise](https://neurips.cc/virtual/2023/poster/70813), which has 29 words.

</details>


## Cool NeurIPS Projects

| **Title** | **Paper** | **Code** | **Project Page** | **Hugging Face** |
|:---------:|:---------:|:--------:|:----------------:|:----------------:|
| An Inverse Scaling Law for CLIP Training | [![arXiv](https://img.shields.io/badge/arXiv-2305.07017-b31b1b.svg)](https://arxiv.org/abs/2305.07017) | [![GitHub](https://img.shields.io/github/stars/UCSC-VLAA/CLIPA?style=social)](https://github.com/UCSC-VLAA/CLIPA)|  |  |
| Augmenting Language Models with Long-Term Memory | [![arXiv](https://img.shields.io/badge/arXiv-2306.07174-b31b1b.svg)](https://arxiv.org/abs/2306.07174) | [![GitHub](https://img.shields.io/github/stars/Victorwz/LongMem?style=social)](https://github.com/Victorwz/LongMem)|  |  |
| Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models | [![arXiv](https://img.shields.io/badge/arXiv-2304.09842-b31b1b.svg)](https://arxiv.org/abs/2304.09842) | [![GitHub](https://img.shields.io/github/stars/lupantech/chameleon-llm?style=social)](https://github.com/lupantech/chameleon-llm)| [Project](https://chameleon-llm.github.io/) |  |
| Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models | [![arXiv](https://img.shields.io/badge/arXiv-2305.15023-b31b1b.svg)](https://arxiv.org/abs/2305.15023) | [![GitHub](https://img.shields.io/github/stars/luogen1996/LaVIN?style=social)](https://github.com/luogen1996/LaVIN)| [Project](https://luogen1996.github.io/lavin/) |  |
| DataComp: In search of the next generation of multimodal datasets | [![arXiv](https://img.shields.io/badge/arXiv-2304.14108-b31b1b.svg)](https://arxiv.org/abs/2304.14108) | [![GitHub](https://img.shields.io/github/stars/mlfoundations/datacomp?style=social)](https://github.com/mlfoundations/datacomp)| [Project](https://www.datacomp.ai/) |  |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | [![arXiv](https://img.shields.io/badge/arXiv-2305.18290-b31b1b.svg)](https://arxiv.org/abs/2305.18290) | [![GitHub](https://img.shields.io/github/stars/eric-mitchell/direct-preference-optimization?style=social)](https://github.com/eric-mitchell/direct-preference-optimization)|  |  |
| DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data | [![arXiv](https://img.shields.io/badge/arXiv-2306.09344-b31b1b.svg)](https://arxiv.org/abs/2306.09344) | [![GitHub](https://img.shields.io/github/stars/ssundaram21/dreamsim?style=social)](https://github.com/ssundaram21/dreamsim)| [Project](https://dreamsim-nights.github.io/) |  |
| Fine-Tuning Language Models with Just Forward Passes | [![arXiv](https://img.shields.io/badge/arXiv-2305.17333-b31b1b.svg)](https://arxiv.org/abs/2305.17333) | [![GitHub](https://img.shields.io/github/stars/princeton-nlp/MeZO?style=social)](https://github.com/princeton-nlp/MeZO)|  |  |
| Generating Images with Multimodal Language Models | [![arXiv](https://img.shields.io/badge/arXiv-2305.17216-b31b1b.svg)](https://arxiv.org/abs/2305.17216) | [![GitHub](https://img.shields.io/github/stars/kohjingyu/gill?style=social)](https://github.com/kohjingyu/gill)| [Project](https://jykoh.com/gill) |  |
| Holistic Evaluation of Text-To-Image Models | [![arXiv](https://img.shields.io/badge/arXiv-2311.04287-b31b1b.svg)](https://arxiv.org/abs/2311.04287) | [![GitHub](https://img.shields.io/github/stars/stanford-crfm/heim?style=social)](https://github.com/stanford-crfm/heim)| [Project](https://crfm.stanford.edu/heim/latest/) |  |
| HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face | [![arXiv](https://img.shields.io/badge/arXiv-2303.17580-b31b1b.svg)](https://arxiv.org/abs/2303.17580) | [![GitHub](https://img.shields.io/github/stars/microsoft/JARVIS?style=social)](https://github.com/microsoft/JARVIS)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/microsoft/HuggingGPT) |
| ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation | [![arXiv](https://img.shields.io/badge/arXiv-2304.05977-b31b1b.svg)](https://arxiv.org/abs/2304.05977) | [![GitHub](https://img.shields.io/github/stars/THUDM/ImageReward?style=social)](https://github.com/THUDM/ImageReward)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/THUDM/ImageReward) |
| InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning | [![arXiv](https://img.shields.io/badge/arXiv-2305.06500-b31b1b.svg)](https://arxiv.org/abs/2305.06500) | [![GitHub](https://img.shields.io/github/stars/salesforce/LAVIS/tree/main/projects/instructblip?style=social)](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](Salesforce/instructblip-vicuna-7b) |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | [![arXiv](https://img.shields.io/badge/arXiv-2306.05685-b31b1b.svg)](https://arxiv.org/abs/2306.05685) | [![GitHub](https://img.shields.io/github/stars/lm-sys/FastChat/tree/main/fastchat/llm_judge?style=social)](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)|  |  |
| LAMM: Multi-Modal Large Language Models and Applications as AI Agents | [![arXiv](https://img.shields.io/badge/arXiv-2306.06687-b31b1b.svg)](https://arxiv.org/abs/2306.06687) | [![GitHub](https://img.shields.io/github/stars/OpenGVLab/LAMM?style=social)](https://github.com/OpenGVLab/LAMM)| [Project](https://openlamm.github.io/) |  |
| LIMA: Less Is More for Alignment | [![arXiv](https://img.shields.io/badge/arXiv-2305.11206-b31b1b.svg)](https://arxiv.org/abs/2305.11206) | |  |  |
| LLM-Pruner: On the Structural Pruning of Large Language Models | [![arXiv](https://img.shields.io/badge/arXiv-2305.11627-b31b1b.svg)](https://arxiv.org/abs/2305.11627) | [![GitHub](https://img.shields.io/github/stars/horseee/LLM-Pruner?style=social)](https://github.com/horseee/LLM-Pruner)|  |  |
| LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenario | [![arXiv](https://img.shields.io/badge/arXiv-2310.08348-b31b1b.svg)](https://arxiv.org/abs/2310.08348) | [![GitHub](https://img.shields.io/github/stars/opendilab/LightZero?style=social)](https://github.com/opendilab/LightZero)|  |  |
| MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion | [![arXiv](https://img.shields.io/badge/arXiv-2307.01097-b31b1b.svg)](https://arxiv.org/abs/2307.01097) | [![GitHub](https://img.shields.io/github/stars/Tangshitao/MVDiffusion?style=social)](https://github.com/Tangshitao/MVDiffusion)| [Project](https://mvdiffusion.github.io/) |  |
| MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing | [![arXiv](https://img.shields.io/badge/arXiv-2306.10012-b31b1b.svg)](https://arxiv.org/abs/2306.10012) | [![GitHub](https://img.shields.io/github/stars/OSU-NLP-Group/MagicBrush?style=social)](https://github.com/OSU-NLP-Group/MagicBrush)| [Project](https://osu-nlp-group.github.io/MagicBrush/) | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/datasets/osunlp/MagicBrush) |
| Mathematical Capabilities of ChatGPT | [![arXiv](https://img.shields.io/badge/arXiv-2301.13867-b31b1b.svg)](https://arxiv.org/abs/2301.13867) | [![GitHub](https://img.shields.io/github/stars/friederrr/GHOSTS?style=social)](https://github.com/friederrr/GHOSTS)| [Project](https://ghosts.friederrr.org/) |  |
| Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation | [![arXiv](https://img.shields.io/badge/arXiv-2306.17115-b31b1b.svg)](https://arxiv.org/abs/2306.17115) | [![GitHub](https://img.shields.io/github/stars/NeuralCarver/Michelangelo?style=social)](https://github.com/NeuralCarver/Michelangelo)| [Project](https://neuralcarver.github.io/michelangelo/) |  |
| Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset | [![arXiv](https://img.shields.io/badge/arXiv-2307.00818-b31b1b.svg)](https://arxiv.org/abs/2307.00818) | [![GitHub](https://img.shields.io/github/stars/IDEA-Research/Motion-X?style=social)](https://github.com/IDEA-Research/Motion-X)| [Project](https://motion-x-dataset.github.io/) |  |
| MotionGPT: Human Motion as Foreign Language | [![arXiv](https://img.shields.io/badge/arXiv-2306.14795-b31b1b.svg)](https://arxiv.org/abs/2306.14795) | [![GitHub](https://img.shields.io/github/stars/OpenMotionLab/MotionGPT?style=social)](https://github.com/OpenMotionLab/MotionGPT)| [Project](https://motion-gpt.github.io/) | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/OpenMotionLab/MotionGPT) |
| OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents | [![arXiv](https://img.shields.io/badge/arXiv-2306.16527-b31b1b.svg)](https://arxiv.org/abs/2306.16527) | [![GitHub](https://img.shields.io/github/stars/huggingface/OBELICS?style=social)](https://github.com/huggingface/OBELICS)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) |
| Photoswap: Personalized Subject Swapping in Images | [![arXiv](https://img.shields.io/badge/arXiv-2305.18286-b31b1b.svg)](https://arxiv.org/abs/2305.18286) | [![GitHub](https://img.shields.io/github/stars/eric-ai-lab/photoswap?style=social)](https://github.com/eric-ai-lab/photoswap)| [Project](https://photoswap.github.io/) |  |
| Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation | [![arXiv](https://img.shields.io/badge/arXiv-2305.01569-b31b1b.svg)](https://arxiv.org/abs/2305.01569) | [![GitHub](https://img.shields.io/github/stars/yuvalkirstain/PickScore?style=social)](https://github.com/yuvalkirstain/PickScore)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) |
| QLoRA: Efficient Finetuning of Quantized LLMs | [![arXiv](https://img.shields.io/badge/arXiv-2305.14314-b31b1b.svg)](https://arxiv.org/abs/2305.14314) | [![GitHub](https://img.shields.io/github/stars/artidoro/qlora?style=social)](https://github.com/artidoro/qlora)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi) |
| Reflexion: Language Agents with Verbal Reinforcement Learning | [![arXiv](https://img.shields.io/badge/arXiv-2303.11366-b31b1b.svg)](https://arxiv.org/abs/2303.11366) | [![GitHub](https://img.shields.io/github/stars/noahshinn/reflexion?style=social)](https://github.com/noahshinn/reflexion)|  |  |
| ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting | [![arXiv](https://img.shields.io/badge/arXiv-2307.12348-b31b1b.svg)](https://arxiv.org/abs/2307.12348) | [![GitHub](https://img.shields.io/github/stars/zsyOAOA/ResShift?style=social)](https://github.com/zsyOAOA/ResShift)| [Project](https://zsyoaoa.github.io/projects/resshift/) |  |
| Segment Anything in 3D with NeRFs | [![arXiv](https://img.shields.io/badge/arXiv-2304.12308-b31b1b.svg)](https://arxiv.org/abs/2304.12308) | [![GitHub](https://img.shields.io/github/stars/Jumpat/SegmentAnythingin3D?style=social)](https://github.com/Jumpat/SegmentAnythingin3D)| [Project](https://jumpat.github.io/SA3D/) |  |
| Segment Anything in High Quality | [![arXiv](https://img.shields.io/badge/arXiv-2306.01567-b31b1b.svg)](https://arxiv.org/abs/2306.01567) | [![GitHub](https://img.shields.io/github/stars/SysCV/sam-hq?style=social)](https://github.com/SysCV/sam-hq)|  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/sam-hq-team/sam-hq) |
| Segment Everything Everywhere All at Once | [![arXiv](https://img.shields.io/badge/arXiv-2304.06718-b31b1b.svg)](https://arxiv.org/abs/2304.06718) | [![GitHub](https://img.shields.io/github/stars/UX-Decoder/Segment-Everything-Everywhere-All-At-Once?style=social)](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)|  |  |
| Self-Refine: Iterative Refinement with Self-Feedback | [![arXiv](https://img.shields.io/badge/arXiv-2303.17651-b31b1b.svg)](https://arxiv.org/abs/2303.17651) | [![GitHub](https://img.shields.io/github/stars/madaan/self-refine?style=social)](https://github.com/madaan/self-refine)| [Project](https://selfrefine.info/) |  |
| Simple and Controllable Music Generation | [![arXiv](https://img.shields.io/badge/arXiv-2306.05284-b31b1b.svg)](https://arxiv.org/abs/2306.05284) | [![GitHub](https://img.shields.io/github/stars/facebookresearch/audiocraft?style=social)](https://github.com/facebookresearch/audiocraft)|  |  |
| Squeeze, Recover and Relabel : Dataset Condensation at ImageNet Scale From A New Perspective | [![arXiv](https://img.shields.io/badge/arXiv-2306.13092-b31b1b.svg)](https://arxiv.org/abs/2306.13092) | [![GitHub](https://img.shields.io/github/stars/VILA-Lab/SRe2L?style=social)](https://github.com/VILA-Lab/SRe2L)|  |  |
| The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only | [![arXiv](https://img.shields.io/badge/arXiv-2306.01116-b31b1b.svg)](https://arxiv.org/abs/2306.01116) | |  | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) |
| Toolformer: Language Models Can Teach Themselves to Use Tools | [![arXiv](https://img.shields.io/badge/arXiv-2302.04761-b31b1b.svg)](https://arxiv.org/abs/2302.04761) | |  |  |
| Unlimiformer: Long-Range Transformers with Unlimited Length Input | [![arXiv](https://img.shields.io/badge/arXiv-2305.01625-b31b1b.svg)](https://arxiv.org/abs/2305.01625) | [![GitHub](https://img.shields.io/github/stars/abertsch72/unlimiformer?style=social)](https://github.com/abertsch72/unlimiformer)|  |  |
| Visual Instruction Tuning | [![arXiv](https://img.shields.io/badge/arXiv-2304.08485-b31b1b.svg)](https://arxiv.org/abs/2304.08485) | [![GitHub](https://img.shields.io/github/stars/haotian-liu/LLaVA?style=social)](https://github.com/haotian-liu/LLaVA)| [Project](https://llava-vl.github.io/) | [![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/badayvedat/LLaVA) |

## Conference Schedule

Note: GitHub automatically truncates files larger than 512 KB. This means that papers in session 6 may not render correctly on GitHub. Please download the file and open it locally to view the full schedule.

<details><summary><h3 style='display: inline;'> Poster Session 1: Tuesday, Dec 12, 08:45 CT</h3></summary>

### (S)GD over Diagonal Linear Networks: Implicit bias, Large Stepsizes and Edge of Stability

**Authors:** Mathieu Even, Scott Pesme, Suriya Gunasekar, Nicolas Flammarion

### 3D Indoor Instance Segmentation in an Open-World

**Authors:** Mohamed El Amine Boudjoghra, Salwa Al Khatib, Jean Lahoud, Hisham Cholakkal, Rao Anwer, Salman Khan, Fahad Shahbaz Khan

### A Batch-to-Online Transformation under Random-Order Model

**Authors:** Jing Dong, Yuichi Yoshida

### A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking

**Authors:** Hao Yan, Chaozhuo Li, Ruosong Long, Chao Yan, Jianan Zhao, Wenwen Zhuang, Jun Yin, Peiyan Zhang, Weihao Han, Hao Sun, Weiwei Deng, Qi Zhang, Lichao Sun, Xing Xie, Senzhang Wang

### A Fast and Accurate Estimator for Large Scale Linear Model via Data Averaging

**Authors:** Rui Wang, Yanyan Ouyang, Yu Panpan, Wangli Xu

### A Guide Through the Zoo of Biased SGD

**Authors:** Yury Demidovich, Grigory Malinovsky, Igor Sokolov, Peter Richtarik

### A High-Resolution Dataset for Instance Detection with Multi-View Object Capture

**Authors:** QIANQIAN SHEN, Yunhan Zhao, Nahyun Kwon, Jeeeun Kim, Yanan Li, Shu Kong

### A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship

**Authors:** Shiyu Hu, Dailing Zhang, wu meiqi, Xiaokun Feng, Xuchen Li, Xin Zhao, Kaiqi Huang

### A Path to Simpler Models Starts With Noise

**Authors:** Lesia Semenova, Harry Chen, Ronald Parr, Cynthia Rudin

### A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning

**Authors:** Valeriia Cherepanova, Roman Levin, Gowthami Somepalli, Jonas Geiping, C. Bayan Bruss, Andrew Wilson, Tom Goldstein, Micah Goldblum

### A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs

**Authors:** Xingyue Huang, Miguel Romero, Ismail Ceylan, Pablo Barceló

### A Theory of Unsupervised Translation Motivated by Understanding Animal Communication

**Authors:** Shafi Goldwasser, David Gruber, Adam Tauman Kalai, Orr Paradise

### [Oral] A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning

**Authors:** Alicia Curth, Alan Jeffares, Mihaela van der Schaar

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1D

### A Unified Approach for Maximizing Continuous DR-submodular Functions

**Authors:** Mohammad Pedramfar, Christopher Quinn, Vaneet Aggarwal

### A Unified Discretization Framework for Differential Equation Approach with Lyapunov Arguments for Convex Optimization

**Authors:** Kansei Ushiyama, Shun Sato, Takayasu Matsuo

### A Unifying Perspective on Multi-Calibration: Game Dynamics for Multi-Objective Learning

**Authors:** Nika Haghtalab, Michael Jordan, Eric Zhao

### A unified framework for information-theoretic generalization bounds

**Authors:** Yifeng Chu, Maxim Raginsky

### ALGO: Synthesizing Algorithmic Programs with Generated Oracle Verifiers

**Authors:** Kexun Zhang, Danqing Wang, Jingtao Xia, William Yang Wang, Lei Li

### AND: Adversarial Neural Degradation for Learning Blind Image Super-Resolution

**Authors:** Fangzhou Luo, Xiaolin Wu, Yanhui Guo

### ANTN: Bridging Autoregressive Neural Networks and Tensor Networks for Quantum Many-Body Simulation

**Authors:** Zhuo Chen, Laker Newhouse, Eddie Chen, Di Luo, Marin Soljacic

### [Spotlight] AbDiffuser: full-atom generation of in-vitro functioning antibodies

**Authors:** Karolis Martinkus, Jan Ludwiczak, WEI-CHING LIANG, Julien Lafrance-Vanasse, Isidro Hotzel, Arvind Rajpal, Yan Wu, Kyunghyun Cho, Richard Bonneau, Vladimir Gligorijevic, Andreas Loukas

### AbdomenAtlas-8K: Annotating 8,000 CT Volumes for Multi-Organ Segmentation in Three Weeks

**Authors:** Chongyu Qu, Tiezheng Zhang, Hualin Qiao, jie liu, Yucheng Tang, Alan Yuille, Zongwei Zhou

### [Oral] Abide by the law and follow the flow: conservation laws for gradient flows

**Authors:** Sibylle Marcotte, Remi Gribonval, Gabriel Peyré

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1D

### Accelerated Zeroth-order Method for Non-Smooth Stochastic Convex Optimization Problem with Infinite Variance

**Authors:** Nikita Kornilov, Ohad Shamir, Aleksandr Lobanov, Darina Dvinskikh, Alexander Gasnikov, Innokentiy Shibaev, Eduard Gorbunov, Samuel Horváth

### Accelerating Exploration with Unlabeled Prior Data

**Authors:** Qiyang Li, Jason Zhang, Dibya Ghosh, Amy Zhang, Sergey Levine

### Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples

**Authors:** Hao Sun, Alihan Hüyük, Daniel Jarrett, Mihaela van der Schaar

### AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking

**Authors:** Yang Yu, Qi Liu, Kai Zhang, Yuren Zhang, Chao Song, Min Hou, Yuqing Yuan, Zhihao Ye, ZAIXI ZHANG, Sanshi Lei Yu

### Adaptive Test-Time Personalization for Federated Learning

**Authors:** Wenxuan Bao, Tianxin Wei, Haohan Wang, Jingrui He

### Add and Thin: Diffusion for Temporal Point Processes

**Authors:** David Lüdke, Marin Biloš, Oleksandr Shchur, Marten Lienen, Stephan Günnemann

### Advancing Bayesian Optimization via Learning Correlated Latent Space

**Authors:** Seunghun Lee, Jaewon Chu, Sihyeon Kim, Juyeon Ko, Hyunwoo Kim

### Adversarial Attacks on Online Learning to Rank with Click Feedback

**Authors:** Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, Adam Wierman

### Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Linear Subspaces

**Authors:** Odelia Melamed, Gilad Yehudai, Gal Vardi

### Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions

**Authors:** Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

### Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors

**Authors:** Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi

### Agnostically Learning Single-Index Models using Omnipredictors

**Authors:** Aravind Gollakota, Parikshit Gopalan, Adam Klivans, Konstantinos Stavropoulos

### AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling

**Authors:** Sachin Chauhan, Zeel Bharatkumar Patel, Sayan Ranu, Rijurekha Sen, Nipun Batra

### AircraftVerse: A Large-Scale Multimodal Dataset of Aerial Vehicle Designs

**Authors:** Adam Cobb, Anirban Roy, Daniel Elenius, Frederick Heim, Brian Swenson, Sydney Whittington, James Walker, Theodore Bapty, Joseph Hite, Karthik Ramani, Christopher McComb, Susmit Jha

### Aligning Gradient and Hessian for Neural Signed Distance Function

**Authors:** Ruian Wang, Zixiong Wang, Yunxiao Zhang, Shuangmin Chen, Shiqing Xin, Changhe Tu, Wenping Wang

### [Spotlight] Aligning Synthetic Medical Images with Clinical Knowledge using Human Feedback

**Authors:** Shenghuan Sun, Greg Goldgof, Atul Butte, Ahmed Alaa

### AllSim: Simulating and Benchmarking Resource Allocation Policies in Multi-User Systems

**Authors:** Jeroen Berrevoets, Daniel Jarrett, Alex Chan, Mihaela van der Schaar

### Alpha-divergence Variational Inference Meets Importance Weighted Auto-Encoders: Methodology and Asymptotics

**Authors:** Kamélia Daudel, Joe Benton, Yuyang Shi, Arnaud Doucet

### [Spotlight] Alternating Updates for Efficient Transformers

**Authors:** Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, Xin Wang

### Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation

**Authors:** Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing Lu, Zhengyang Wang, Ruirui Li, Zhen Li, Monica Cheng, Rahul Goutam, Haiyang Zhang, Karthik Subbian, Suhang Wang, Yizhou Sun, Jiliang Tang, Bing Yin, Xianfeng Tang

### Amortized Reparametrization: Efficient and Scalable Variational Inference for Latent SDEs

**Authors:** Kevin Course, Prasanth Nair

### An Inverse Scaling Law for CLIP Training

**Authors:** Xianhang Li, Zeyu Wang, Cihang Xie

### Analysis of Variance of Multiple Causal Networks

**Authors:** Zhongli Jiang, Dabao Zhang

### Analyzing Vision Transformers for Image Classification in Class Embedding Space

**Authors:** Martina G. Vilas, Timothy Schaumlöffel, Gemma Roig

### Anonymous Learning via Look-Alike Clustering: A Precise Analysis of Model Generalization

**Authors:** Adel Javanmard, Vahab Mirrokni

### Approximate inference of marginals using the IBIA framework

**Authors:** Shivani Bathla, Vinita Vasudevan

### Approximately Equivariant Graph Networks

**Authors:** Ningyuan Huang, Ron Levie, Soledad Villar

### Arbitrarily Scalable Environment Generators via Neural Cellular Automata

**Authors:** Yulun Zhang, Matthew Fontaine, Varun Bhatt, Stefanos Nikolaidis, Jiaoyang Li

### Are GATs Out of Balance?

**Authors:** Nimrah Mustafa, Aleksandar Bojchevski, Rebekka Burkholz

### Are These the Same Apple? Comparing Images Based on Object Intrinsics

**Authors:** Klemen Kotar, Stephen Tian, Hong-Xing Yu, Dan Yamins, Jiajun Wu

### Are Vision Transformers More Data Hungry Than Newborn Visual Systems?

**Authors:** Lalit Pandey, Samantha Wood, Justin Wood

### Asymmetric Certified Robustness via Feature-Convex Neural Networks

**Authors:** Samuel Pfrommer, Brendon Anderson, Julien Piet, Somayeh Sojoudi

### Augmentation-free Dense Contrastive Distillation for Efficient Semantic Segmentation

**Authors:** Jiawei Fan, Chao Li, Xiaolong Liu, Meina Song, Anbang Yao

### Automatic Integration for Spatiotemporal Neural Point Processes

**Authors:** Zihao Zhou, Rose Yu

### Bayesian Learning of Optimal Policies in Markov Decision Processes with Countably Infinite State-Space

**Authors:** Saghar Adler, Vijay Subramanian

### Bayesian Optimization with Cost-varying Variable Subsets

**Authors:** Sebastian Tay, Chuan Sheng Foo, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low

### [Spotlight] Bayesian target optimisation for high-precision holographic optogenetics

**Authors:** Marcus Triplett, Marta Gajowa, Hillel Adesnik, Liam Paninski

### [Spotlight] Behavior Alignment via Reward Function Optimization

**Authors:** Dhawal Gupta, Yash Chandak, Scott Jordan, Philip Thomas, Bruno da Silva

### Beta Diffusion

**Authors:** Mingyuan Zhou, Tianqi Chen, Zhendong Wang, Huangjie Zheng

### Better Private Linear Regression Through Better Private Feature Selection

**Authors:** Travis Dick, Jennifer Gillenwater, Matthew Joseph

### Binarized Neural Machine Translation

**Authors:** Yichi Zhang, Ankush Garg, Yuan Cao, Lukasz Lew, Behrooz Ghorbani, Zhiru Zhang, Orhan Firat

### Binary Classification with Confidence Difference

**Authors:** Wei Wang, Lei Feng, Yuchen Jiang, Gang Niu, Min-Ling Zhang, Masashi Sugiyama

### Black-box Backdoor Defense via Zero-shot Image Purification

**Authors:** Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, Ninghao Liu

### Blurred-Dilated Method for Adversarial Attacks

**Authors:** Yang Deng, Weibin Wu, Jianping Zhang, Zibin Zheng

### BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information

**Authors:** Mehran Kazemi, Quan Yuan, Deepti Bhatia, Najoung Kim, Xin Xu, Vaiva Imbrasaite, Deepak Ramachandran

### Boosting with Tempered Exponential Measures

**Authors:** Richard Nock, Ehsan Amid, Manfred Warmuth

### Bottleneck Structure in Learned Features: Low-Dimension vs Regularity Tradeoff

**Authors:** Arthur Jacot

### Bounded rationality in structured  density estimation

**Authors:** Tianyuan Teng, Kevin Li, Hang Zhang

### Bounding the Invertibility of Privacy-preserving Instance Encoding using Fisher Information

**Authors:** Kiwan Maeng, Chuan Guo, Sanjay Kariyappa, G. Edward Suh

### BubbleML: A Multiphase Multiphysics Dataset and Benchmarks for Machine Learning

**Authors:** Sheikh Md Shakeel Hassan, Arthur Feeney, Akash Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu, Yoonjin Won, Aparna Chandramowlishwaran

### CAP:  Correlation-Aware Pruning for Highly-Accurate Sparse Vision Models

**Authors:** Denis Kuznedelev, Eldar Kurtić, Elias Frantar, Dan Alistarh

### CAT-Walk: Inductive Hypergraph Learning via Set Walks

**Authors:** Ali Behrouz, Farnoosh Hashemi, Sadaf Sadeghian, Margo Seltzer

### [Spotlight] CLIP-OGD: An Experimental Design for Adaptive Neyman Allocation in Sequential Experiments

**Authors:** Jessica Dai, Paula Gradu, Christopher Harshaw

### CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models

**Authors:** Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng LYU, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf

### COOM: A Game Benchmark for Continual Reinforcement Learning

**Authors:** Tristan Tomilin, Meng Fang, Yudi Zhang, Mykola Pechenizkiy

### CORNN: Convex optimization of recurrent neural networks for rapid inference of neural dynamics

**Authors:** Fatih Dinc, Adam Shai, Mark Schnitzer, Hidenori Tanaka

### CP-SLAM: Collaborative Neural Point-based SLAM System

**Authors:** Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, Zhaopeng Cui

### CSOT: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels

**Authors:** Wanxing Chang, Ye Shi, Jingya Wang

### [Spotlight] Can semi-supervised learning use all the data effectively? A lower bound perspective

**Authors:** Alexandru Tifrea, Gizem Yüce, Amartya Sanyal, Fanny Yang

### Causal Context Connects Counterfactual Fairness to Robust Prediction and Group Fairness

**Authors:** Jacy Anthis, Victor Veitch

### Cause-Effect Inference in Location-Scale Noise Models: Maximum Likelihood vs. Independence Testing

**Authors:** Xiangyu Sun, Oliver Schulte

### Censored Sampling of Diffusion Models Using 3 Minutes of Human Feedback

**Authors:** TaeHo Yoon, Kibeom Myoung, Keon Lee, Jaewoong Cho, Albert No, Ernest Ryu

### Certification of Distributional Individual Fairness

**Authors:** Matthew Wicker, Vihari Piratla, Adrian Weller

### [Oral] Characteristic Circuits

**Authors:** Zhongjie Yu, Martin Trapp, Kristian Kersting

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1C

### Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond

**Authors:** Oleg Platonov, Denis Kuznedelev, Artem Babenko, Liudmila Prokhorenkova

### Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models

**Authors:** Deepak Narayanan, Keshav Santhanam, Peter Henderson, Rishi Bommasani, Tony Lee, Percy Liang

### ChessGPT: Bridging Policy Learning and Language Modeling

**Authors:** Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, Jun Wang

### CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data

**Authors:** Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue

### Classical Simulation of Quantum Circuits: Parallel Environments and Benchmark

**Authors:** Xiao-Yang Liu, Zeliang Zhang

### ClimateSet: A Large-Scale Climate Model Dataset for Machine Learning

**Authors:** Julia Kaltenborn, Charlotte Lange, Venkatesh Ramesh, Philippe Brouillard, Yaniv Gurwicz, Chandni Nagda, Jakob Runge, Peer Nowack, David Rolnick

### Cola: A Benchmark for Compositional Text-to-image Retrieval

**Authors:** Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan Plummer, Ranjay Krishna, Kate Saenko

### Collapsed Inference for Bayesian Deep Learning

**Authors:** Zhe Zeng, Guy Van den Broeck

### Computational Guarantees for Doubly Entropic Wasserstein Barycenters

**Authors:** Tomas Vaskevicius, Lénaïc Chizat

### ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image

**Authors:** Senthil Purushwalkam, Nikhil Naik

### Concept Algebra for (Score-Based) Text-Controlled Generative Models

**Authors:** Zihao Wang, Lin Gui, Jeffrey Negrea, Victor Veitch

### [Spotlight] Conditional independence testing under misspecified inductive biases

**Authors:** Felipe Maia Polo, Yuekai Sun, Moulinath Banerjee

### Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems

**Authors:** Sihan Zeng, Thinh Doan, Justin Romberg

### Conservative Offline Policy Adaptation in Multi-Agent Games

**Authors:** Chengjie Wu, Pingzhong Tang, Jun Yang, Yujing Hu, Tangjie Lv, Changjie Fan, Chongjie Zhang

### Constructing Non-isotropic Gaussian Diffusion Model Using Isotropic Gaussian Diffusion Model for Image Editing

**Authors:** Xi Yu, Xiang Gu, Haozhi Liu, Jian Sun

### Contextual Bandits and Imitation Learning with Preference-Based Active Queries

**Authors:** Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu

### ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling

**Authors:** Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li

### Continuous-Time Functional Diffusion Processes

**Authors:** Giulio Franzese, Giulio Corallo, Simone Rossi, Markus Heinonen, Maurizio Filippone, Pietro Michiardi

### Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning

**Authors:** Siming Lan, Rui Zhang, Qi Yi, Jiaming Guo, Shaohui Peng, Yunkai Gao, Fan Wu, Ruizhi Chen, Zidong Du, Xing Hu, xishan zhang, Ling Li, Yunji Chen

### Coupled Reconstruction of Cortical Surfaces by Diffeomorphic Mesh Deformation

**Authors:** Hao Zheng, Hongming Li, Yong Fan

### Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks

**Authors:** Haoyi Duan, Yan Xia, Zhou Mingze, Li Tang, Jieming Zhu, Zhou Zhao

### CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion

**Authors:** Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang

### Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models

**Authors:** Julien Siems, Konstantin Ditschuneit, Winfried Ripken, Alma Lindborg, Maximilian Schambach, Johannes Otterbach, Martin Genzel

### Customizable Image Synthesis with Multiple Subjects

**Authors:** Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao

### DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis

**Authors:** Youngjoong Kwon, Lingjie Liu, Henry Fuchs, Marc Habermann, Christian Theobalt

### DIFFER:Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning

**Authors:** Xunhan Hu, Jian Zhao, Wengang Zhou, Ruili Feng, Houqiang Li

### DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction

**Authors:** Mohammadreza Pourreza, Davood Rafiei

### DISCS: A Benchmark for Discrete Sampling

**Authors:** Katayoon Goshvadi, Haoran Sun, Xingchao Liu, Azade Nova, Ruqi Zhang, Will Grathwohl, Dale Schuurmans, Hanjun Dai

### DVSOD: RGB-D Video Salient Object Detection

**Authors:** Jingjing Li, Wei Ji, Size Wang, Wenbo Li, Li cheng

### Data Market Design through Deep Learning

**Authors:** Sai Srivatsa Ravindranath, Yanchen Jiang, David Parkes

### Data-Driven Network Neuroscience: On Data Collection and Benchmark

**Authors:** Jiaxing Xu, Yunhan Yang, David Huang, Sophi Shilpa Gururajapathy, Yiping Ke, Miao Qiao, Alan Wang, Haribalan Kumar, Josh McGeown, Eryn Kwon

### Data-Informed Geometric Space Selection

**Authors:** Shuai Zhang, Wenqi Jiang

### DataPerf: Benchmarks for Data-Centric AI Development

**Authors:** Mark Mazumder, Colby Banbury, Xiaozhe Yao, Bojan Karlaš, William Gaviria Rojas, Sudnya Diamos, Greg Diamos, Lynn He, Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Will Cukierski, Juan Ciro, Lora Aroyo, Bilge Acun, Lingjiao Chen, Mehul Raje, Max Bartolo, Evan Sabri Eyuboglu, Amirata Ghorbani, Emmett Goodman, Addison Howard, Oana Inel, Tariq Kane, Christine R. Kirkpatrick, D. Sculley, Tzu-Sheng Kuo, Jonas Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen Paritosh, Ce Zhang, James Zou, Carole-Jean Wu, Cody Coleman, Andrew Ng, Peter Mattson, Vijay Janapa Reddi

### Decentralized Matrix Sensing: Statistical Guarantees and Fast Convergence

**Authors:** Marie Maros, Gesualdo Scutari

### [Oral] DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models

**Authors:** Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, Bo Li

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1B

### Deductive Verification of Chain-of-Thought Reasoning

**Authors:** Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang, Mingu Lee, Roland Memisevic, Hao Su

### Deep Contract Design via Discontinuous Networks

**Authors:** Tonghan Wang, Paul Duetting, Dmitry Ivanov, Inbal Talgam-Cohen, David Parkes

### Deep Optimal Transport: A Practical Algorithm for Photo-realistic Image Restoration

**Authors:** Theo Adrai, Guy Ohayon, Michael Elad, Tomer Michaeli

### Degraded Polygons Raise Fundamental Questions of Neural Network Perception

**Authors:** Leonard Tang, Dan Ley

### [Spotlight] Demystifying Oversmoothing in Attention-Based Graph Neural Networks

**Authors:** Xinyi Wu, Amir Ajorlou, Zihui Wu, Ali Jadbabaie

### Density of States Prediction of Crystalline Materials via Prompt-guided Multi-Modal Transformer

**Authors:** Namkyeong Lee, Heewoong Noh, Sungwon Kim, Dongmin Hyun, Gyoung S. Na, Chanyoung Park

### DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models

**Authors:** XiMing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, Dong Xu

### Differentiable Neuro-Symbolic Reasoning on Large-Scale Knowledge Graphs

**Authors:** CHEN SHENGYUAN, Yunfeng Cai, Huang Fang, Xiao Huang, Mingming Sun

### Differentiable Sampling of Categorical Distributions Using the CatLog-Derivative Trick

**Authors:** Lennert De Smet, Emanuele Sansone, Pedro Zuidberg Dos Martires

### Differentiable sorting for censored time-to-event data.

**Authors:** Andre Vauvelle, Benjamin Wild, Roland Eils, Spiros Denaxas

### [Spotlight] Differentially Private Approximate Near Neighbor Counting in High Dimensions

**Authors:** Alexandr Andoni, Piotr Indyk, Sepideh Mahabadi, Shyam Narayanan

### [Spotlight] Differentially Private Image Classification by Learning Priors from Random Processes

**Authors:** Xinyu Tang, Ashwinee Panda, Vikash Sehwag, Prateek Mittal

### Diffusion Probabilistic Models for Structured Node Classification

**Authors:** Hyosoon Jang, Seonghyun Park, Sangwoo Mo, Sungsoo Ahn

### Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability

**Authors:** Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

### Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection

**Authors:** Cheng-Ju Ho, Chen-Hsuan Tai, Yen-Yu Lin, Ming-Hsuan Yang, Yi-Hsuan Tsai

### Diplomat: A Dialogue Dataset for Situated PragMATic Reasoning

**Authors:** Hengli Li, Song-Chun Zhu, Zilong Zheng

### Directed Cyclic Graph for Causal Discovery from Multivariate Functional Data

**Authors:** Saptarshi Roy, Raymond K. W. Wong, Yang Ni

### Dis-inhibitory neuronal circuits can control the sign of synaptic plasticity

**Authors:** Julian Rossbroich, Friedemann Zenke

### Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier

**Authors:** Yuling Yao, Justin Domke

### Disentangled Counterfactual Learning for Physical Audiovisual Commonsense Reasoning

**Authors:** Changsheng Lv, Shuai Zhang, Yapeng Tian, Mengshi Qi, Huadong Ma

### Disentangled Wasserstein Autoencoder for T-Cell Receptor Engineering

**Authors:** Tianxiao Li, Hongyu Guo, Filippo Grazioli, Mark Gerstein, Martin Renqiang Min

### Disentangling Cognitive Diagnosis with Limited Exercise Labels

**Authors:** Xiangzhi Chen, Le Wu, Fei Liu, Lei Chen, Kun Zhang, Richang Hong, Meng Wang

### [Spotlight] Distribution-Free Statistical Dispersion Control for Societal Applications

**Authors:** Zhun Deng, Thomas Zollo, Jake Snell, Toniann Pitassi, Richard Zemel

### [Spotlight] Distributionally Robust Linear Quadratic Control

**Authors:** Bahar Taskesen, Dan Iancu, Çağıl Koçyiğit, Daniel Kuhn

### Diverse Conventions for Human-AI Collaboration

**Authors:** Bidipta Sarkar, Andy Shih, Dorsa Sadigh

### DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method

**Authors:** Ahmed Khaled, Konstantin Mishchenko, Chi Jin

### [Spotlight] Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models

**Authors:** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun

### Domain Agnostic Fourier Neural Operators

**Authors:** Ning Liu, Siavash Jafarzadeh, Yue Yu

### Don’t Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner

**Authors:** Zhengxiang Shi, Aldo Lipani

### Don’t blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy

**Authors:** Aahlad Manas Puli, Lily Zhang, Yoav Wald, Rajesh Ranganath

### DreamSparse: Escaping from Plato’s Cave with 2D Diffusion Model Given Sparse Views

**Authors:** Paul Yoo, Jiaxian Guo, Yutaka Matsuo, Shixiang (Shane) Gu

### DreamWaltz: Make a Scene with Complex 3D Animatable Avatars

**Authors:** Yukun Huang, Jianan Wang, Ailing Zeng, He CAO, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, Lei Zhang

### Dual Mean-Teacher: An Unbiased Semi-Supervised Framework for Audio-Visual Source Localization

**Authors:** Yuxin Guo, Shijie Ma, Hu Su, Zhiqing Wang, Yuhao Zhao, Wei Zou, Siyang Sun, Yun Zheng

### DynaDojo: An Extensible Platform for Benchmarking Scaling in Dynamical System Identification

**Authors:** Logan M Bhamidipaty, Tommy Bruzzese, Caryn Tran, Rami Ratl Mrad, Maxinder S. Kanwal

### Dynamic Pricing and Learning with Bayesian Persuasion

**Authors:** Shipra Agrawal, Yiding Feng, Wei Tang

### Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies

**Authors:** Michael Beukman, Devon Jarvis, Richard Klein, Steven James, Benjamin Rosman

### EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images

**Authors:** Seongsu Bae, Daeun Kyung, Jaehee Ryu, Eunbyeol Cho, Gyubok Lee, Sunjun Kweon, Jungwoo Oh, Lei Ji, Eric Chang, Tackeun Kim, Edward Choi

### EMBERSim: A Large-Scale Databank for Boosting Similarity Search in Malware Analysis

**Authors:** Dragos Georgian Corlatescu, Alexandru Dinu, Mihaela Petruta Gaman, Paul Sumedrea

### Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

**Authors:** Steven Adriaensen, Herilalaina Rakotoarison, Samuel Müller, Frank Hutter

### Efficient Learning of Linear Graph Neural Networks via Node Subsampling

**Authors:** Seiyun Shin, Ilan Shomorony, Han Zhao

### Efficient Neural Music Generation

**Authors:** Max W. Y. Lam, Qiao Tian, Tang Li, Zongyu Yin, Siyuan Feng, Ming Tu, Yuliang Ji, Rui Xia, Mingbo Ma, Xuchen Song, Jitong Chen, Wang Yuping, Yuxuan Wang

### Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents

**Authors:** wonje choi, Woo Kyung Kim, SeungHyun Kim, Honguk Woo

### Efficient Subgame Refinement for Extensive-form Games

**Authors:** Zhenxing Ge, Zheng Xu, Tianyu Ding, Wenbin Li, Yang Gao

### Egocentric Planning for Scalable Embodied Task Achievement

**Authors:** Xiatoian Liu, Hector Palacios, Christian Muise

### Eliciting User Preferences for Personalized Multi-Objective Decision Making through Comparative Feedback

**Authors:** Han Shao, Lee Cohen, Avrim Blum, Yishay Mansour, Aadirupa Saha, Matthew Walter

### Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples

**Authors:** Shashanka Venkataramanan, Ewa Kijak, laurent amsaleg, Yannis Avrithis

### End-To-End Latent Variational Diffusion Models for Inverse Problems in High Energy Physics

**Authors:** Alexander Shmakov, Kevin Greif, Michael Fenton, Aishik Ghosh, Pierre Baldi, Daniel Whiteson

### Energy-Based Sliced Wasserstein Distance

**Authors:** Khai Nguyen, Nhat Ho

### Enhancing Adversarial Robustness via Score-Based Optimization

**Authors:** Boya Zhang, Weijian Luo, Zhihua Zhang

### Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics

**Authors:** Koen Minartz, Yoeri Poels, Simon Koop, Vlado Menkovski

### Equivariant Spatio-Temporal Attentive Graph Networks to Simulate Physical Dynamics

**Authors:** Liming Wu, Zhichao Hou, Jirui Yuan, Yu Rong, Wenbing Huang

### Equivariant flow matching

**Authors:** Leon Klein, Andreas Krämer, Frank Noe

### Estimating Causal Effects Identifiable from a Combination of Observations and Experiments

**Authors:** Yonghan Jung, Ivan Diaz, Jin Tian, Elias Bareinboim

### Estimating Noise Correlations Across Continuous Conditions With Wishart Processes

**Authors:** Amin Nejatbakhsh, Isabel Garon, Alex Williams

### Euler-Lagrange Analysis of Generative Adversarial Networks

**Authors:** Siddarth Asokan, Chandra Seelamantula

### Evaluating Self-Supervised Learning for Molecular Graph Embeddings

**Authors:** Hanchen Wang, Jean Kaddour, Shengchao Liu, Jian Tang, Joan Lasenby, Qi Liu

### Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance

**Authors:** Jonathan Crabbé, Mihaela van der Schaar

### [Oral] Exact Bayesian Inference on Discrete Models via Probability Generating Functions: A Probabilistic Programming Approach

**Authors:** Fabian Zaiser, Andrzej Murawski, Luke Ong

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1C

### Expanding Small-Scale Datasets with Guided Imagination

**Authors:** Yifan Zhang, Daquan Zhou, Bryan Hooi, Kai Wang, Jiashi Feng

### [Spotlight] Explaining the Uncertain: Stochastic Shapley Values for Gaussian Process Models

**Authors:** Siu Lun Chau, Krikamol Muandet, Dino Sejdinovic

### Exploring Why Object Recognition Performance Degrades Across Income Levels and Geographies with Factor Annotations

**Authors:** Laura Gustafson, Megan Richards, Melissa Hall, Caner Hazirbas, Diane Bouchacourt, Mark Ibrahim

### Exponentially Convergent Algorithms for Supervised Matrix Factorization

**Authors:** Joowon Lee, Hanbaek Lyu, Weixin Yao

### [Spotlight] Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model

**Authors:** Yule Wang, Zijing Wu, Chengrui Li, Anqi Wu

### FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation

**Authors:** Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, Lu Hou

### Face Reconstruction from Facial Templates by Learning Latent Space of a Generator Network

**Authors:** Hatef Otroshi Shahreza, Sébastien Marcel

### FaceComposer: A Unified Model for Versatile Facial Content Creation

**Authors:** Jiayu Wang, Kang Zhao, Yifeng Ma, Shiwei Zhang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou

### Facing Off World Model Backbones: RNNs, Transformers, and S4

**Authors:** Fei Deng, Junyeong Park, Sungjin Ahn

### Failure-Aware Gaussian Process Optimization with Regret Bounds

**Authors:** Shogo Iwazaki, Shion Takeno, Tomohiko Tanabe, Mitsuru Irie

### Fair Allocation of Indivisible Chores: Beyond Additive Costs

**Authors:** Bo Li, Fangxiao Wang, Yu Zhou

### Fair Streaming Principal Component Analysis: Statistical and Algorithmic Viewpoint

**Authors:** Junghyun Lee, Hanseul Cho, Se-Young Yun, Chulhee Yun

### [Spotlight] Faith and Fate: Limits of Transformers on Compositionality

**Authors:** Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang (Lorraine) Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena Hwang, Soumya Sanyal, Xiang Ren, Allyson Ettinger, Zaid Harchaoui, Yejin Choi

### Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training

**Authors:** Aleksandra Nowak, Bram Grooten, Decebal Constantin Mocanu, Jacek Tabor

### Fast Asymptotically Optimal Algorithms for Non-Parametric Stochastic Bandits

**Authors:** Dorian Baudry, Fabien Pesquerel, Rémy Degenne, Odalric-Ambrym Maillard

### Fast Conditional Mixing of MCMC Algorithms for Non-log-concave Distributions

**Authors:** Xiang Cheng, Bohan Wang, Jingzhao Zhang, Yusong Zhu

### [Spotlight] Feature Adaptation for Sparse Linear Regression

**Authors:** Jonathan Kelner, Frederic Koehler, Raghu Meka, Dhruv Rohatgi

### Feature Dropout: Revisiting the Role of Augmentations in Contrastive Learning

**Authors:** Alex Tamkin, Margalit Glasgow, Xiluo He, Noah Goodman

### FedL2P: Federated Learning to Personalize

**Authors:** Royson Lee, Minyoung Kim, Da Li, Xinchi Qiu, Timothy Hospedales, Ferenc Huszar, Nicholas Lane

### Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator

**Authors:** Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, Yu Zhang

### Fine-grained Expressivity of Graph Neural Networks

**Authors:** Jan Böker, Ron Levie, Ningyuan Huang, Soledad Villar, Christopher Morris

### Finite-Time Analysis of Whittle Index based Q-Learning for Restless Multi-Armed Bandits with Neural Network Function Approximation

**Authors:** GUOJUN XIONG, Jian Li

### FlatMatch: Bridging Labeled Data and Unlabeled Data with Cross-Sharpness for Semi-Supervised Learning

**Authors:** Zhuo Huang, Li Shen, Jun Yu, Bo Han, Tongliang Liu

### ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning

**Authors:** Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, Mingsheng Long

### Fragment-based Pretraining and Finetuning on Molecular Graphs

**Authors:** Kha-Dinh Luong, Ambuj K Singh

### Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization

**Authors:** Kamil Dreczkowski, Antoine Grosnit, Haitham Bou Ammar

### FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models

**Authors:** Lihe Yang, Xiaogang Xu, Bingyi Kang, Yinghuan Shi, Hengshuang Zhao

### Function Space Bayesian Pseudocoreset for Bayesian Neural Networks

**Authors:** Balhae Kim, Hyungi Lee, Juho Lee

### [Spotlight] GLIME: General, Stable and Local LIME Explanation

**Authors:** Zeren Tan, Yang Tian, Jian Li

### GNeSF: Generalizable Neural Semantic Fields

**Authors:** Hanlin Chen, Chen Li, Mengqi Guo, Zhiwen Yan, Gim Hee Lee

### GSLB: The Graph Structure Learning Benchmark

**Authors:** Zhixun Li, Liang Wang, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, Jeffrey Yu

### Gaussian Differential Privacy on Riemannian Manifolds

**Authors:** Yangdi Jiang, Xiaotian Chang, Yi Liu, Lei Ding, Linglong Kong, Bei Jiang

### GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image

**Authors:** Mingjian Zhu, Hanting Chen, Qiangyu YAN, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang

### GenS: Generalizable Neural Surface Reconstruction from Multi-View Images

**Authors:** Rui Peng, Xiaodong Gu, Luyang Tang, Shihe Shen, Fanqi Yu, Ronggang Wang

### Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations

**Authors:** Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang

### Generalizable One-shot 3D Neural Head Avatar

**Authors:** Xueting Li, Shalini De Mello, Sifei Liu, Koki Nagano, Umar Iqbal, Jan Kautz

### Geometry-Informed Neural Operator for Large-Scale 3D PDEs

**Authors:** Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Animashree Anandkumar

### Glance and Focus: Memory Prompting for Multi-Event Video Question Answering

**Authors:** Ziyi Bai, Ruiping Wang, Xilin Chen

### Global Identifiability of  $\ell_1$-based Dictionary Learning via Matrix Volume Optimization

**Authors:** Jingzhou Hu, Kejun Huang

### Global Optimality and Finite Sample Analysis of Softmax Off-Policy Actor Critic under State Distribution Mismatch

**Authors:** Shangtong Zhang, Remi Tachet des Combes, Romain Laroche

### Global Structure-Aware Diffusion Process for Low-light Image Enhancement

**Authors:** Jinhui HOU, Zhiyu Zhu, Junhui Hou, Hui LIU, Huanqiang Zeng, Hui Yuan

### Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces

**Authors:** Martin Ryner, Jan Kronqvist, Johan Karlsson

### [Spotlight] GloptiNets: Scalable Non-Convex Optimization with Certificates

**Authors:** Gaspard Beugnot, Julien Mairal, Alessandro Rudi

### GlyphControl: Glyph Conditional Control for Visual Text Generation

**Authors:** Yukang Yang, Dongnan Gui, YUHUI YUAN, Weicong Liang, Haisong Ding, Han Hu, Kai Chen

### GlyphControl: Glyph Conditional Controllable Visual Text Generation

**Authors:** Yukang Yang, Dongnan Gui, YUHUI YUAN, Weicong Liang, Haisong Ding, Han Hu, Kai Chen

### Gradient Descent with Linearly Correlated Noise: Theory and Applications to Differential Privacy

**Authors:** Anastasiia Koloskova, Ryan McKenna, Zachary Charles, John Rush, H. Brendan McMahan

### Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents

**Authors:** Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, brian ichter

### Guide Your Agent with Adaptive Multimodal Rewards

**Authors:** Changyeon Kim, Younggyo Seo, Hao Liu, Lisa Lee, Jinwoo Shin, Honglak Lee, Kimin Lee

### H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang "Atlas" Wang, Beidi Chen

### HA-ViD: A Human Assembly Video Dataset for Comprehensive Assembly Knowledge Understanding

**Authors:** Hao Zheng, Regina Lee, Yuqian Lu

### HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception

**Authors:** Junkun Yuan, Xinyu Zhang, Hao Zhou, Jian Wang, Zhongwei Qiu, Zhiyin Shao, Shaofeng Zhang, Sifan Long, Kun Kuang, Kun Yao, Junyu Han, Errui Ding, Lanfen Lin, Fei Wu, Jingdong Wang

### HT-Step: Aligning Instructional Articles with How-To Videos

**Authors:** Triantafyllos Afouras, Effrosyni Mavroudi, Tushar Nagarajan, Huiyu Wang, Lorenzo Torresani

### Harnessing Hard Mixed Samples with Decoupled Regularizer

**Authors:** Zicheng Liu, Siyuan Li, Ge Wang, Lirong Wu, Cheng Tan, Stan Z. Li

### Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks

**Authors:** Jimmy Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

### Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning

**Authors:** Yangru Huang, Peixi Peng, Yifan Zhao, Haoran Xu, Mengyue Geng, Yonghong Tian

### Hierarchical Open-vocabulary Universal Image Segmentation

**Authors:** Xudong Wang, Shufan Li, Konstantinos Kallidromitis, Yusuke Kato, Kazuki Kozuka, Trevor Darrell

### Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration

**Authors:** Longlin Yu, Tianyu Xie, Yu Zhu, Tong Yang, Xiangyu Zhang, Cheng Zhang

### High-dimensional Contextual Bandit Problem without Sparsity

**Authors:** Junpei Komiyama, Masaaki Imaizumi

### Holistic Evaluation of Text-to-Image Models

**Authors:** Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, Percy Liang

### Holistic Transfer: Towards Non-Disruptive Fine-Tuning with Partial Target Data

**Authors:** Cheng-Hao Tu, Hong-You Chen, Zheda Mai, Jike Zhong, Vardaan Pahuja, Tanya Berger-Wolf, Song Gao, Charles Stewart, Yu Su, Wei-Lun (Harry) Chao

### How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources

**Authors:** Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah Smith, Iz Beltagy, Hannaneh Hajishirzi

### How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization

**Authors:** Hai Zhang, Hang Yu, Junqiao Zhao, Di Zhang, xiao zhang, Hongtu Zhou, Chang Huang, Chen Ye

### [Oral] How to Turn Your Knowledge Graph Embeddings into Generative Models

**Authors:** Lorenzo Loconte, Nicola Di Mauro, Robert Peharz, Antonio Vergari

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1C

### HubRouter: Learning Global Routing via Hub Generation and Pin-hub Connection

**Authors:** Xingbo Du, Chonghua Wang, Ruizhe Zhong, Junchi Yan

### Human-in-the-Loop Optimization for Deep Stimulus Encoding in Visual Prostheses

**Authors:** Jacob Granley, Tristan Fauvel, Matthew Chalk, Michael Beyeler

### IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL

**Authors:** Pascal Leroy, Pablo G. Morato, Jonathan Pisane, Athanasios Kolios, Damien Ernst

### [Spotlight] Imitation Learning from Imperfection: Theoretical Justifications and Algorithms

**Authors:** Ziniu Li, Tian Xu, Zeyu Qin, Yang Yu, Zhi-Quan Luo

### [Spotlight] Implicit Bias of Gradient Descent for Logistic Regression at the Edge of Stability

**Authors:** Jingfeng Wu, Vladimir Braverman, Jason Lee

### Implicit Contrastive Representation Learning with Guided Stop-gradient

**Authors:** Byeongchan Lee, Sehyun Lee

### Improved Best-of-Both-Worlds Guarantees for Multi-Armed Bandits: FTRL with General Regularizers and Multiple Optimal Arms

**Authors:** Tiancheng Jin, Junyan Liu, Haipeng Luo

### Improved Communication Efficiency in Federated Natural Policy Gradient via ADMM-based Gradient Updates

**Authors:** Guangchen Lan, Han Wang, James Anderson, Christopher Brinton, Vaneet Aggarwal

### Improving Adversarial Robustness via Information Bottleneck Distillation

**Authors:** Huafeng Kuang, Hong Liu, Yongjian Wu, Shin'ichi Satoh, Rongrong Ji

### Improving Diffusion-Based Image Synthesis with Context Prediction

**Authors:** Ling Yang, Jingwei Liu, Shenda Hong, Zhilong Zhang, Zhilin Huang, Zheming Cai, Wentao Zhang, Bin CUI

### Improving multimodal datasets with image captioning

**Authors:** Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, Ludwig Schmidt

### Information Design in Multi-Agent Reinforcement Learning

**Authors:** Yue Lin, Wenhao Li, Hongyuan Zha, Baoxiang Wang

### Inserting Anybody in Diffusion Models via Celeb Basis

**Authors:** Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, Huicheng Zheng

### Intensity Profile Projection: A Framework for Continuous-Time Representation Learning for Dynamic Networks

**Authors:** Alexander Modell, Ian Gallagher, Emma Ceccherini, Nick Whiteley, Patrick Rubin-Delanchy

### Interaction Measures, Partition Lattices and Kernel Tests for High-Order Interactions

**Authors:** Zhaolu Liu, Robert Peach, Pedro A.M Mediano, Mauricio Barahona

### Interpretability at Scale: Identifying Causal Mechanisms in Alpaca

**Authors:** Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, Noah Goodman

### Intervention Generalization: A View from Factor Graph Models

**Authors:** Gecia Bravo-Hermsdorff, David Watson, Jialin Yu, Jakob Zeitler, Ricardo Silva

### Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective

**Authors:** João Carvalho, Mengtao Zhang, Robin Geyer, Carlos Cotrini, Joachim M Buhmann

### Investigating how ReLU-networks encode symmetries

**Authors:** Georg Bökman, Fredrik Kahl

### [Spotlight] Is Learning in Games Good for the Learners?

**Authors:** William Brown, Jon Schneider, Kiran Vodrahalli

### Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network

**Authors:** Tristan Deleu, Mizu Nishikawa-Toomey, Jithendaraa Subramanian, Nikolay Malkin, Laurent Charlin, Yoshua Bengio

### Joint Prompt Optimization of Stacked LLMs using Variational Inference

**Authors:** Alessandro Sordoni, Eric Yuan, Marc-Alexandre Côté, Matheus Pereira, Adam Trischler, Ziang Xiao, Arian Hosseini, Friederike Niedtner, Nicolas Le Roux

### JourneyDB: A Benchmark for Generative Image Understanding

**Authors:** Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, Hongsheng Li

### Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, Ion Stoica

### Katakomba: Tools and Benchmarks for Data-Driven NetHack

**Authors:** Vladislav Kurenkov, Alexander Nikulin, Denis Tarasov, Sergey Kolesnikov

### Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization

**Authors:** Clement Benard, Brian Staber, Sébastien Da Veiga

### Kernelized Reinforcement Learning with Order Optimal Regret Bounds

**Authors:** Sattar Vakili, Julia Olkhovskaya

### [Spotlight] Kiki or Bouba? Sound Symbolism in Vision-and-Language Models

**Authors:** Morris Alper, Hadar Averbuch-Elor

### Kissing to Find a Match: Efficient Low-Rank Permutation Representation

**Authors:** Hannah Dröge, Zorah Lähner, Yuval Bahat, Onofre Martorell Nadal, Felix Heide, Michael Moeller

### Knowledge Diffusion for Distillation

**Authors:** Tao Huang, Yuan Zhang, Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Chang Xu

### Koopman Kernel Regression

**Authors:** Petar Bevanda, Max Beier, Armin Lederer, Stefan Sosnowski, Eyke Hüllermeier, Sandra Hirche

### KuaiSim: A Comprehensive Simulator for Recommender Systems

**Authors:** Kesen Zhao, Shuchang Liu, Qingpeng Cai, Xiangyu Zhao, Ziru Liu, Dong Zheng, Peng Jiang, Kun Gai

### L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference

**Authors:** Julia Linhart, Alexandre Gramfort, Pedro Rodrigues

### LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

**Authors:** Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

### Language Model Alignment with Elastic Reset

**Authors:** Michael Noukhovitch, Samuel Lavoie, Florian Strub, Aaron Courville

### Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment

**Authors:** Hao Liu, Wilson Yan, Pieter Abbeel

### Language Semantic Graph Guided Data-Efficient Learning

**Authors:** Wenxuan Ma, Shuang Li, lincan Cai, Jingxuan Kang

### Large sample spectral analysis of graph-based multi-manifold clustering

**Authors:** Nicolas Garcia Trillos, Pengfei He, Chenghui Li

### LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting

**Authors:** Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, LEI BAI, Chao Huang, Zhenguang Liu, Bryan Hooi, Roger Zimmermann

### LayoutPrompter: Awaken the Design Ability of Large Language Models

**Authors:** Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, Dongmei Zhang

### [Oral] LeanDojo: Theorem Proving with Retrieval-Augmented Language Models

**Authors:** Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J Prenger, Animashree Anandkumar

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1B

### Learning Adaptive Tensorial Density Fields for Clean Cryo-ET Reconstruction

**Authors:** Yuanhao Wang, Ramzi Idoughi, Wolfgang Heidrich

### Learning Curves for Deep Structured Gaussian Feature Models

**Authors:** Jacob Zavatone-Veth, Cengiz Pehlevan

### Learning DAGs from Data with Few Root Causes

**Authors:** Panagiotis Misiakos, Chris Wendler, Markus Püschel

### Learning Domain-Aware Detection Head with Prompt Tuning

**Authors:** Haochen Li, Rui Zhang, Hantao Yao, Xinkai Song, Yifan Hao, Yongwei Zhao, Ling Li, Yunji Chen

### Learning Efficient Surrogate Dynamic Models with Graph Spline Networks

**Authors:** Chuanbo Hua, Federico Berto, Michael Poli, Stefano Massaroli, Jinkyoo Park

### Learning From Biased Soft Labels

**Authors:** Hua Yuan, Yu Shi, Ning Xu, Xu Yang, Xin Geng, Yong Rui

### [Spotlight] Learning Generalizable Agents via Saliency-guided Features Decorrelation

**Authors:** Sili Huang, Yanchao Sun, Jifeng Hu, Siyuan Guo, Hechang Chen, Yi Chang, Lichao Sun, Bo Yang

### Learning Mixtures of Gaussians Using the DDPM Objective

**Authors:** Kulin Shah, Sitan Chen, Adam Klivans

### Learning Rate Free Bayesian Inference in Constrained Domains

**Authors:** Louis Sharrock, Lester Mackey, Christopher Nemeth

### Learning Shared Safety Constraints from Multi-task Demonstrations

**Authors:** Konwoo Kim, Gokul Swamy, ZUXIN LIU, DING ZHAO, Sanjiban Choudhury, Steven Wu

### Learning World Models with Identifiable Factorization

**Authors:** Yuren Liu, Biwei Huang, Zhengmao Zhu, Honglong Tian, Mingming Gong, Yang Yu, Kun Zhang

### Learning better with Dale’s Law: A Spectral Perspective

**Authors:** Pingsheng Li, Jonathan Cornford, Arna Ghosh, Blake Richards

### [Spotlight] Learning from Active Human Involvement through Proxy Value Propagation

**Authors:** Zhenghao Peng, Wenjie Mo, Chenda Duan, Quanyi Li, Bolei Zhou

### Learning from Visual Observation via Offline Pretrained State-to-Go Transformer

**Authors:** Bohan Zhou, Ke Li, Jiechuan Jiang, Zongqing Lu

### Learning to Parameterize Visual Attributes for Open-set Fine-grained Retrieval

**Authors:** Shijie Wang, Jianlong Chang, Haojie Li, Zhihui Wang, Wanli Ouyang, Qi Tian

### Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt

**Authors:** Yining Ma, Zhiguang Cao, Yeow Meng Chee

### Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition

**Authors:** Yuhang Zhang, Yaqi Li, lixiong Qin, Xuannan Liu, Weihong Deng

### LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models

**Authors:** Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Aditya K, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John Nay, Jonathan Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, Zehua Li

### LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios

**Authors:** Yazhe Niu, YUAN PU, Zhenjie Yang, Xueyan Li, Tong Zhou, Jiyuan Ren, Shuai Hu, Hongsheng Li, Yu Liu

### LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference

**Authors:** Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding

### [Spotlight] List and Certificate Complexities in Replicable Learning

**Authors:** Peter Dixon, A. Pavan, Jason Vander Woude, N. V. Vinodchandran

### LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning

**Authors:** Atsuyuki Miyai, Qing Yu, Go Irie, Kiyoharu Aizawa

### LoRA: A Logical Reasoning Augmented Dataset for Visual Question Answering

**Authors:** Jingying Gao, Qi Wu, Alan Blair, Maurice Pagnucco

### Locality-Aware Generalizable Implicit Neural Representation

**Authors:** Doyup Lee, Chiheon Kim, Minsu Cho, WOOK SHIN HAN

### LogSpecT: Feasible Graph Learning Model from Stationary Signals with Recovery Guarantees

**Authors:** Shangyuan LIU, Linglingzhi Zhu, Anthony Man-Cho So

### Logarithmic-Regret Quantum Learning Algorithms for Zero-Sum Games

**Authors:** Minbo Gao, Zhengfeng Ji, Tongyang Li, Qisheng Wang

### Long Sequence Hopfield Memory

**Authors:** Hamza Chaudhry, Jacob Zavatone-Veth, Dmitry Krotov, Cengiz Pehlevan

### Lovász Principle for Unsupervised Graph Representation Learning

**Authors:** Ziheng Sun, Chris Ding, Jicong Fan

### M5HisDoc: A Large-scale Multi-style Chinese Historical Document Analysis Benchmark

**Authors:** Yongxin Shi, Chongyu Liu, Dezhi Peng, Cheng Jian, Jiarong Huang, Lianwen Jin

### MAViL: Masked Audio-Video Learners

**Authors:** Po-Yao Huang, Vasu Sharma, Hu Xu, Chaitanya Ryali, haoqi fan, Yanghao Li, Shang-Wen Li, Gargi Ghosh, Jitendra Malik, Christoph Feichtenhofer

### MCUFormer: Deploying Vision Tranformers on Microcontrollers with Limited Memory

**Authors:** Yinan Liang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, Jiwen Lu

### MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection

**Authors:** Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho

### MIMEx: Intrinsic Rewards from Masked Input Modeling

**Authors:** Toru Lin, Allan Jabri

### Markovian Sliced Wasserstein Distances: Beyond Independent Projections

**Authors:** Khai Nguyen, Tongzheng Ren, Nhat Ho

### Mathematical Capabilities of ChatGPT

**Authors:** Simon Frieder, Luca Pinchetti, Chevalier, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, Julius Berner

### Matrix Compression via Randomized Low Rank and Low Precision Factorization

**Authors:** Rajarshi Saha, Varun Srivastava, Mert Pilanci

### Maximum State Entropy Exploration using Predecessor and Successor Representations

**Authors:** Arnav Kumar Jain, Lucas Lehnert, Irina Rish, Glen Berseth

### May the Force be with You: Unified Force-Centric Pre-Training for 3D Molecular Conformations

**Authors:** Rui Feng, Qi Zhu, Huan Tran, Binghong Chen, Aubrey Toland, Rampi Ramprasad, Chao Zhang

### Metis: Understanding and Enhancing In-Network Regular Expressions

**Authors:** Zhengxin Zhang, Yucheng Huang, Guanglin Duan, Qing Li, Dan Zhao, Yong Jiang, Lianbo Ma, Xi Xiao, Hengyang Xu

### Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation

**Authors:** Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, BIN FU, Tao Chen, Gang Yu, Shenghua Gao

### Mitigating Source Bias for Fairer Weak Supervision

**Authors:** Changho Shin, Sonia Cromp, Dyah Adila, Frederic Sala

### Mixed Samples as Probes for Unsupervised Model Selection in Domain Adaptation

**Authors:** Dapeng Hu, Jian Liang, Jun Hao Liew, Chuhui Xue, Song Bai, Xinchao Wang

### MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks

**Authors:** Allen Nie, Yuhui Zhang, Atharva Shailesh Amdekar, Chris Piech, Tatsunori Hashimoto, Tobias Gerstenberg

### Mode Connectivity in Auction Design

**Authors:** Christoph Hertrich, Yixin Tao, László A. Végh

### [Spotlight] Model Sparsity Can Simplify Machine Unlearning

**Authors:** jinghan jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, PRANAY SHARMA, Sijia Liu

### Model and Feature Diversity for Bayesian Neural Networks in Mutual Learning

**Authors:** Van Cuong Pham, Cuong C Nguyen, Trung Le, Dinh Phung, Gustavo Carneiro, Thanh-Toan Do

### Model-Based Control with Sparse Neural Dynamics

**Authors:** Ziang Liu, Genggeng Zhou, Jeff He, Tobia Marcucci, Fei-Fei Li, Jiajun Wu, Yunzhu Li

### Model-Free Active Exploration in Reinforcement Learning

**Authors:** Alessio Russo, Alexandre Proutiere

### Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder

**Authors:** Michael Bereket, Theofanis Karaletsos

### Module-wise Adaptive Distillation for Multimodality Foundation Models

**Authors:** Chen Liang, Jiahui Yu, Ming-Hsuan Yang, Matthew Brown, Yin Cui, Tuo Zhao, Boqing Gong, Tianyi Zhou

### Moment Matching Denoising Gibbs Sampling

**Authors:** Mingtian Zhang, Alex Hawkins-Hooker, Brooks Paige, David Barber

### Monte Carlo Tree Search with Boltzmann Exploration

**Authors:** Michael Painter, Mohamed Baioumy, Nick Hawes, Bruno Lacerda

### Mr. HiSum: A Large-scale Dataset for Video Highlight Detection and Summarization

**Authors:** Jinhwan Sul, Jihoon Han, Joonseok Lee

### Multi-Swap k-Means++

**Authors:** Lorenzo Beretta, Vincent Cohen-Addad, Silvio Lattanzi, Nikos Parotsidis

### Multi-task Representation Learning for Pure Exploration in Bilinear Bandits

**Authors:** Subhojyoti Mukherjee, Qiaomin Xie, Josiah Hanna, Robert Nowak

### Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text

**Authors:** Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, Yejin Choi

### Multimodal Deep Learning Model Unveils Behavioral Dynamics of V1 Activity in Freely Moving Mice

**Authors:** Aiwen Xu, Yuchen Hou, Cristopher Niell, Michael Beyeler

### Near Optimal Reconstruction of Spherical Harmonic Expansions

**Authors:** Amir Zandieh, Insu Han, Haim Avron

### Near-Optimal Bounds for Learning Gaussian Halfspaces with Random Classification Noise

**Authors:** Ilias Diakonikolas, Jelena Diakonikolas, Daniel Kane, Puqian Wang, Nikos Zarifis

### Neural Fields with Hard Constraints of Arbitrary Differential Order

**Authors:** Fangcheng Zhong, Kyle Fogarty, Param Hanji, Tianhao Wu, Alejandro Sztrajman, Andrew Spielberg, Andrea Tagliasacchi, Petra Bosilj, Cengiz Oztireli

### Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features

**Authors:** Mingli Zhu, Shaokui Wei, Hongyuan Zha, Baoyuan Wu

### NeuroEvoBench:  Benchmarking Evolutionary Optimizers for Deep Learning Applications

**Authors:** Robert Lange, Yujin Tang, Yingtao Tian

### [Spotlight] Newton–Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems

**Authors:** Lingbing Guo, Weiqing Wang, Zhuo Chen, Ningyu Zhang, Zequn Sun, Yixuan Lai, Qiang Zhang, Huajun Chen

### No-Regret Learning with Unbounded Losses: The Case of Logarithmic Pooling

**Authors:** Eric Neyman, Tim Roughgarden

### Non-Convex Bilevel Optimization with Time-Varying Objective Functions

**Authors:** Sen Lin, Daouda Sow, Kaiyi Ji, Yingbin Liang, Ness Shroff

### Norm-based Generalization Bounds for Sparse Neural Networks

**Authors:** Tomer Galanti, Mengjia Xu, Liane Galanti, Tomaso Poggio

### NurViD: A Large Expert-Level Video Database for Nursing Procedure Activity Understanding

**Authors:** Ming Hu, Lin Wang, Siyuan Yan, Don Ma, Qingli Ren, Peng Xia, Wei Feng, Peibo Duan, Lie Ju, Zongyuan Ge

### ODE-based Recurrent Model-free Reinforcement Learning for POMDPs

**Authors:** Xuanle Zhao, Duzhen Zhang, Han Liyuan, Tielin Zhang, Bo Xu

### OceanBench: The Sea Surface Height Edition

**Authors:** J. Emmanuel Johnson, Quentin Febvre, Anastasiia Gorbunova, Sam Metref, Maxime Ballarotta, Julien Le Sommer, ronan fablet

### Offline Imitation Learning with Variational Counterfactual Reasoning

**Authors:** Zexu Sun, Bowei He, Jinxin Liu, Xu Chen, Chen Ma, Shuai Zhang

### Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization

**Authors:** Xiangsen Wang, Haoran Xu, Yinan Zheng, Xianyuan Zhan

### Offline Reinforcement Learning with Differential Privacy

**Authors:** Dan Qiao, Yu-Xiang Wang

### On Differentially Private Sampling from Gaussian and Product Distributions

**Authors:** Badih Ghazi, Xiao Hu, Ravi Kumar, Pasin Manurangsi

### On Sparse Modern Hopfield Model

**Authors:** Jerry Yao-Chieh Hu, Donglin Yang, Dennis Wu, Chenwei Xu, Bo-Yu Chen, Han Liu

### On skip connections and normalisation layers in deep optimisation

**Authors:** Lachlan MacDonald, Jack Valmadre, Hemanth Saratchandran, Simon Lucey

### On the Convergence and Sample Complexity Analysis of Deep Q-Networks with $\epsilon$-Greedy Exploration

**Authors:** Shuai Zhang, Hongkang Li, Meng Wang, Miao Liu, Pin-Yu Chen, Songtao Lu, Sijia Liu, Keerthiram Murugesan, Subhajit Chaudhury

### On the Convergence to a Global Solution of Shuffling-Type Gradient Algorithms

**Authors:** Lam Nguyen, Trang H. Tran

### On the Generalization Properties of Diffusion Models

**Authors:** Puheng Li, Zhong Li, Huishuai Zhang, Jiang Bian

### On the Identifiability and Interpretability of Gaussian Process Models

**Authors:** Jiawen Chen, Wancen Mu, Yun Li, Didong Li

### On the Interplay between Social Welfare and Tractability of Equilibria

**Authors:** Ioannis Anagnostides, Tuomas Sandholm

### [Spotlight] On the Minimax Regret for Online Learning with Feedback Graphs

**Authors:** Khaled Eldowa, Emmanuel Esposito, Tom Cesari, Nicolò Cesa-Bianchi

### [Spotlight] On the Planning Abilities of Large Language Models - A Critical Investigation

**Authors:** Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, Subbarao Kambhampati

### On the Power of SVD in the Stochastic Block Model

**Authors:** Xinyu Mao, Jiapeng Zhang

### On the Sublinear Regret of GP-UCB

**Authors:** Justin Whitehouse, Aaditya Ramdas, Steven Wu

### One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization

**Authors:** Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, Hao Su

### One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning

**Authors:** Zichang Liu, Zhaozhuo Xu, Benjamin Coleman, Anshumali Shrivastava

### Online Convex Optimization with Unbounded Memory

**Authors:** Raunak Kumar, Sarah Dean, Robert Kleinberg

### Online Map Vectorization for Autonomous Driving: A Rasterization Perspective

**Authors:** Gongjie Zhang, Jiahao Lin, Shuang Wu, yilin song, Zhipeng Luo, Yang Xue, Shijian Lu, Zuoguan Wang

### Online Nonstochastic Model-Free Reinforcement Learning

**Authors:** Udaya Ghai, Arushi Gupta, Wenhan Xia, Karan Singh, Elad Hazan

### Online PCA in Converging Self-consistent Field Equations

**Authors:** Xihan Li, Xiang Chen, Rasul Tutunov, Haitham Bou Ammar, Lei Wang, Jun Wang

### [Oral] Online RL in Linearly $q^\pi$-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore

**Authors:** Gellert Weisz, András György, Csaba Szepesvari

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1A

### Online learning of long-range dependencies

**Authors:** Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, Joao Sacramento

### [Oral] OpenAssistant Conversations - Democratizing Large Language Model Alignment

**Authors:** Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1B

### OpenDataVal: a Unified Benchmark for Data Valuation

**Authors:** Kevin Jiang, Weixin Liang, James Zou, Yongchan Kwon

### OpenIllumination: A Multi-Illumination Dataset for Inverse Rendering Evaluation on Real Objects

**Authors:** Isabella Liu, Linghao Chen, Ziyang Fu, Liwen Wu, Haian Jin, Zhong Li, Chin Ming Ryan Wong, Yi Xu, Ravi Ramamoorthi, Zexiang Xu, Hao Su

### OpenMask3D: Open-Vocabulary 3D Instance Segmentation

**Authors:** Ayca Takmaz, Elisabetta Fedele, Robert Sumner, Marc Pollefeys, Federico Tombari, Francis Engelmann

### Operator Learning with Neural Fields: Tackling PDEs on General Geometries

**Authors:** Louis Serrano, Lise Le Boudec, Armand Kassaï Koupaï, Thomas X Wang, Yuan Yin, Jean-Noël Vittaut, Patrick Gallinari

### Optimal Convergence Rate for Exact Policy Mirror Descent in Discounted Markov Decision Processes

**Authors:** Emmeran Johnson, Ciara Pike-Burke, Patrick Rebeschini

### Optimal Extragradient-Based Algorithms for Stochastic Variational Inequalities with Separable Structure

**Authors:** Angela Yuan, Chris Junchi Li, Gauthier Gidel, Michael Jordan, Quanquan Gu, Simon Du

### Optimal privacy guarantees for a relaxed threat model: Addressing sub-optimal adversaries in differentially private machine learning

**Authors:** Georgios Kaissis, Alexander Ziller, Stefan Kolek, Anneliese Riess, Daniel Rueckert

### Optimal testing using combined test statistics across independent studies

**Authors:** Lasse Vuursteen, Botond Szabo, Aad van der Vaart, Harry van Zanten

### Optimistic Active Exploration of Dynamical Systems

**Authors:** Bhavya, Lenart Treven, Cansu Sancaktar, Sebastian Blaes, Stelian Coros, Andreas Krause

### Optimistic Exploration in Reinforcement Learning Using Symbolic Model Estimates

**Authors:** Sarath Sreedharan, Michael Katz

### Optimizing over trained GNNs via symmetry breaking

**Authors:** Shiqiang Zhang, Juan Campos, Christian Feldmann, David Walz, Frederik Sandfort, Miriam Mathea, Calvin Tsay, Ruth Misener

### Oracle Complexity of Single-Loop Switching Subgradient Methods for Non-Smooth Weakly Convex Functional Constrained Optimization

**Authors:** Yankun Huang, Qihang Lin

### Order Matters in the Presence of Dataset Imbalance for Multilingual Learning

**Authors:** Dami Choi, Derrick Xin, Hamid Dadkhahi, Justin Gilmer, Ankush Garg, Orhan Firat, Chih-Kuan Yeh, Andrew Dai, Behrooz Ghorbani

### [Oral] Ordering-based Conditions for Global Convergence of Policy Gradient Methods

**Authors:** Jincheng Mei, Bo Dai, Alekh Agarwal, Mohammad Ghavamzadeh, Csaba Szepesvari, Dale Schuurmans

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1A

### Orthogonal Non-negative Tensor Factorization based Multi-view Clustering

**Authors:** Jing Li, Quanxue Gao, QIANQIAN WANG, Ming Yang, Wei Xia

### PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance

**Authors:** Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, Jimin Huang

### PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model

**Authors:** Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, Navdeep Jaitly

### PRIOR: Personalized Prior for Reactivating the Information Overlooked in Federated Learning.

**Authors:** Mingjia Shi, Yuhao Zhou, Kai Wang, Huaizheng Zhang, Shudong Huang, Qing Ye, Jiancheng Lv

### [Spotlight] PRODIGY: Enabling In-context Learning Over Graphs

**Authors:** Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, Jure Leskovec

### PackQViT: Faster Sub-8-bit Vision Transformers via Full and Packed Quantization on the Mobile

**Authors:** Peiyan Dong, LEI LU, Chao Wu, Cheng Lyu, Geng Yuan, Hao Tang, Yanzhi Wang

### ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP

**Authors:** Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang

### [Spotlight] Partial Counterfactual Identification of Continuous Outcomes with a Curvature Sensitivity Model

**Authors:** Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel

### [Spotlight] Participatory Personalization in Classification

**Authors:** Hailey Joren, Chirag Nagpal, Katherine Heller, Berk Ustun

### Path following algorithms for $\ell_2$-regularized $M$-estimation with approximation guarantee

**Authors:** Yunzhang Zhu, Renxiong Liu

### Penguin: Parallel-Packed Homomorphic Encryption for Fast Graph Convolutional Network Inference

**Authors:** Ran Ran, Nuo Xu, Tao Liu, Wei Wang, Gang Quan, Wujie Wen

### Percentile Criterion Optimization in Offline Reinforcement Learning

**Authors:** Cyrus Cousins, Elita Lobo, Marek Petrik, Yair Zick

### Persuading Farsighted Receivers in MDPs: the Power of Honesty

**Authors:** Martino Bernasconi, Matteo Castiglioni, Alberto Marchesi, Mirco Mutti

### [Spotlight] Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication

**Authors:** Chaofei Fan, Nick Hahn, Foram Kamdar, Donald Avansino, Guy Wilson, Leigh Hochberg, Krishna V Shenoy, Jaimie Henderson, Francis Willett

### Policy Optimization for Continuous Reinforcement Learning

**Authors:** HANYANG ZHAO, Wenpin Tang, David Yao

### Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control

**Authors:** Nate Rahn, Pierluca D'Oro, Harley Wiltzer, Pierre-Luc Bacon, Marc Bellemare

### Policy Space Diversity for Non-Transitive Games

**Authors:** Jian Yao, Weiming Liu, Haobo Fu, Yaodong Yang, Stephen McAleer, Qiang Fu, Wei Yang

### Polynomially Over-Parameterized Convolutional Neural Networks Contain Structured Strong Winning Lottery Tickets

**Authors:** Arthur da Cunha, Francesco D'Amore, Natale

### PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones

**Authors:** Thad Starner, Sean Forbes, Matthew So, David Martin, Rohit Sridhar, Gururaj Deshpande, Sam Sepah, Sahir Shahryar, Khushi Bhardwaj, Tyler Kwok, Daksh Sehgal, Saad Hassan, Bill Neubauer, Sofia Vempala, Alec Tan, Jocelyn Heath, Unnathi Kumar, Priyanka Mosur, Tavenner Hall, Rajandeep Singh, Christopher Cui, Glenn Cameron, Sohier Dane, Garrett Tanzer

### PrObeD: Proactive Object Detection Wrapper

**Authors:** Vishal Asnani, Abhinav Kumar, Suya You, Xiaoming Liu

### Practical Contextual Bandits with Feedback Graphs

**Authors:** Mengxiao Zhang, Yuheng Zhang, Olga Vrousgou, Haipeng Luo, Paul Mineiro

### [Spotlight] Practical Sharpness-Aware Minimization Cannot Converge All the Way to Optima

**Authors:** Dongkuk Si, Chulhee Yun

### [Spotlight] Pre-Training Protein Encoder via Siamese Sequence-Structure Diffusion Trajectory Prediction

**Authors:** Zuobai Zhang, Minghao Xu, Aurelie Lozano, Vijil Chenthamarakshan, Payel Das, Jian Tang

### [Spotlight] Precise asymptotic generalization for multiclass classification with overparameterized linear models

**Authors:** David Wu, Anant Sahai

### Precision-Recall Divergence Optimization for Generative Modeling with GANs and Normalizing Flows

**Authors:** Alexandre Verine, Benjamin Negrevergne, Muni Sreenivas Pydi, Yann Chevaleyre

### Predicting mutational effects on protein-protein binding via a side-chain diffusion probabilistic model

**Authors:** Shiwei Liu, Tian Zhu, Milong Ren, Chungong Yu, Dongbo Bu, Haicang Zhang

### Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation

**Authors:** Yingyi Chen, Qinghua Tao, Francesco Tonin, Johan Suykens

### ProBio: A Protocol-guided Multimodal Dataset for Molecular Biology Lab

**Authors:** Jieming Cui, Ziren Gong, Baoxiong Jia, Siyuan Huang, Zilong Zheng, Jianzhu Ma, Yixin Zhu

### [Spotlight] ProPILE: Probing Privacy Leakage in Large Language Models

**Authors:** Siwon Kim, Sangdoo Yun, Hwaran Lee, Martin Gubri, Sungroh Yoon, Seong Joon Oh

### Probabilistic Exponential Integrators

**Authors:** Nathanael Bosch, Philipp Hennig, Filip Tronarp

### Probabilistic Inference in Reinforcement Learning Done Right

**Authors:** Jean Tarbouriech, Tor Lattimore, Brendan O'Donoghue

### Projection-Free Methods for Stochastic Simple Bilevel Optimization with Convex Lower-level Problem

**Authors:** Jincheng Cao, Ruichen Jiang, Nazanin Abolfazli, Erfan Yazdandoost Hamedani, Aryan Mokhtari

### ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design

**Authors:** Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Han Spinner, Nathan Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Yarin Gal, Debora Marks

### Provable Guarantees for Neural Networks via Gradient Feature Learning

**Authors:** Zhenmei Shi, Junyi Wei, Yingyu Liang

### [Spotlight] Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks

**Authors:** Eshaan Nichani, Alex Damian, Jason Lee

### [Spotlight] Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond

**Authors:** Omar Chehab, Aapo Hyvarinen, Andrej Risteski

### Provably (More) Sample-Efficient Offline RL with Options

**Authors:** Xiaoyan Hu, Ho-fung Leung

### Provably Efficient Algorithm for Nonstationary Low-Rank MDPs

**Authors:** Yuan Cheng, Jing Yang, Yingbin Liang

### Provably Efficient Offline Reinforcement Learning in Regular Decision Processes

**Authors:** Roberto Cipollone, Anders Jonsson, Alessandro Ronca, Mohammad Sadegh Talebi

### Provably Safe Reinforcement Learning with Step-wise Violation Constraints

**Authors:** Nuoya Xiong, Yihan Du, Longbo Huang

### Punctuation-level Attack: Single-shot and Single Punctuation Can Fool Text Models

**Authors:** wenqiang wang, Chongyang Du, Tao Wang, Kaihao Zhang, Wenhan Luo, Lin Ma, Wei Liu, Xiaochun Cao

### Quantification of Uncertainty with Adversarial Models

**Authors:** Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, Günter Klambauer, Sepp Hochreiter

### Query-based Temporal Fusion with Explicit Motion for 3D Object Detection

**Authors:** Jinghua Hou, Zhe Liu, dingkang liang, Zhikang Zou, Xiaoqing Ye, Xiang Bai

### RD-Suite: A Benchmark for Ranking Distillation

**Authors:** Zhen Qin, Rolf Jagerman, Rama Kumar Pasumarthi, Honglei Zhuang, He Zhang, Aijun Bai, Kai Hui, Le Yan, Xuanhui Wang

### REASONER: An Explainable Recommendation Dataset with Comprehensive Labeling Ground Truths

**Authors:** Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, Ji-Rong Wen

### RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks

**Authors:** Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, HUI LI, Xiaodong Lin

### RaLEs: a Benchmark for Radiology Language Evaluations

**Authors:** Juan M Zambrano Chaves, Nandita Bhaskhar, Maayane Attias, Jean-Benoit Delbrouck, Daniel Rubin, Andreas Loening, Curtis Langlotz, Akshay Chaudhari

### ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction

**Authors:** Jia Guo, shuai lu, Lize Jia, Weihang Zhang, Huiqi Li

### Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals

**Authors:** Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom Mitchell

### Reconciling Competing Sampling Strategies of Network Embedding

**Authors:** Yuchen Yan, Baoyu Jing, Lihui Liu, Ruijie Wang, Jinning Li, Tarek Abdelzaher, Hanghang Tong

### Recovering Unbalanced Communities in the Stochastic Block Model with Application to Clustering with a Faulty Oracle

**Authors:** Chandra Sekhar Mukherjee, Pan Peng, Jiapeng Zhang

### Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans

**Authors:** Kyowoon Lee, Seongun Kim, Jaesik Choi

### [Spotlight] Regularization properties of adversarially-trained linear regression

**Authors:** Antonio Ribeiro, Dave Zachariah, Francis Bach, Thomas Schön

### Reinforcement Learning with Fast and Forgetful Memory

**Authors:** Steven D Morad, Ryan Kortvelesy, Stephan Liwicki, Amanda Prorok

### [Spotlight] Reinforcement-Enhanced Autoregressive Feature Transformation: Gradient-steered Search in Continuous Space for Postfix Expressions

**Authors:** Dongjie Wang, Meng Xiao, Min Wu, pengfei wang, Yuanchun Zhou, Yanjie Fu

### Reliable Off-Policy Learning for Dosage Combinations

**Authors:** Jonas Schweisthal, Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel

### Renku: a platform for sustainable data science

**Authors:** Rok Roškar, Chandrasekhar Ramakrishnan, Michele Volpi, Fernando Perez-Cruz, Lilian Gasser, Firat Ozdemir, Patrick Paitz, Mohammad Alisafaee, Philipp Fischer, Ralf Grubenmann, Eliza Harris, Tasko Olevski, Carl Remlinger, Luis Salamanca, Elisabet Capon Garcia, Lorenzo Cavazzi, Jakub Chrobasik, Darlin Cordoba Osnas, Alessandro Degano, Jimena Dupre, Wesley Johnson, Eike Kettner, Laura Kinkead, Sean Murphy, Flora Thiebaut, Olivier Verscheure

### Reproducibility Study of ”CartoonX: Cartoon Explanations of Image Classifiers”

**Authors:** Aditya Patra, Sina Taslimi, Luke Chin A Foeng, Pratik Kayal

### Reproducibility study of 'Proto2Proto: Can you recognise the car, the way I do?'

**Authors:** Gerson de Kleuver, David Bikker, Wenhua Hu, Bram Veenman

### [Spotlight] ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting

**Authors:** Zongsheng Yue, Jianyi Wang, Chen Change Loy

### Rethinking Conditional Diffusion Sampling with Progressive Guidance

**Authors:** Anh-Dung Dinh, Daochang Liu, Chang Xu

### Rethinking the Role of Token Retrieval in Multi-Vector Retrieval

**Authors:** Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, Vincent Zhao

### Reverse Engineering Self-Supervised Learning

**Authors:** Ido Ben-Shaul, Ravid Shwartz-Ziv, Tomer Galanti, Shai Dekel, Yann LeCun

### Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective

**Authors:** Yingying Fan, Yu Wu, Bo Du, Yutian Lin

### Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models

**Authors:** Naman Deep Singh, Francesco Croce, Matthias Hein

### Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement

**Authors:** Hui Yuan, Kaixuan Huang, Chengzhuo Ni, Minshuo Chen, Mengdi Wang

### Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards

**Authors:** Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, Matthieu Cord

### Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation

**Authors:** Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu

### Rigorous Runtime Analysis of MOEA/D for Solving Multi-Objective Minimum Weight Base Problems

**Authors:** Anh Viet Do, Aneta Neumann, Frank Neumann, Andrew Sutton

### Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning

**Authors:** James Queeney, Mouhacine Benosman

### RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions

**Authors:** Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit Cottereau, Wei Tsang Ooi

### Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy

**Authors:** Dongmin Park, Seola Choi, Doyoung Kim, Hwanjun Song, Jae-Gil Lee

### [Spotlight] Robust Distributed Learning: Tight Error Bounds and Breakdown Point under Data Heterogeneity

**Authors:** Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, Geovani Rizk

### Robust Learning for Smoothed Online Convex Optimization with Feedback Delay

**Authors:** Pengfei Li, Jianyi Yang, Adam Wierman, Shaolei Ren

### SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection

**Authors:** Daehyun Kim, Sungyong Baik, Tae Hyun Kim

### SEENN: Towards Temporal Spiking Early Exit Neural Networks

**Authors:** Yuhang Li, Tamar Geller, Youngeun Kim, Priyadarshini Panda

### SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation

**Authors:** Zhongang Cai, Wanqi Yin, Ailing Zeng, CHEN WEI, Qingping SUN, Wang Yanjun, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Lei Yang, Ziwei Liu

### SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities

**Authors:** Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty

### SSL4EO-L: Datasets and Foundation Models for Landsat Imagery

**Authors:** Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, Arindam Banerjee

### STXD: Structural and Temporal Cross-Modal Distillation for Multi-View 3D Object Detection

**Authors:** Sujin Jang, Dae Ung Jo, Sung Ju Hwang, Dongwook Lee, Daehyun Ji

### SUPA: A Lightweight Diagnostic Simulator for Machine Learning in Particle Physics

**Authors:** Atul Kumar Sinha, Daniele Paliotta, Bálint Máté, John Raine, Tobias Golling, François Fleuret

### Scalable Transformer for PDE Surrogate Modeling

**Authors:** Zijie Li, Dule Shu, Amir Barati Farimani

### Scalarization for Multi-Task and Multi-Domain Learning at Scale

**Authors:** Amelie Royer, Tijmen Blankevoort, Babak Ehteshami Bejnordi

### Scaling Up Differentially Private LASSO Regularized Logistic Regression via Faster Frank-Wolfe Iterations

**Authors:** Edward Raff, Amol Khanna, Fred Lu

### ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling

**Authors:** Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, Bolei Zhou

### Score-based Data Assimilation

**Authors:** François Rozet, Gilles Louppe

### Selective Sampling and Imitation Learning via Online Regression

**Authors:** Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu

### Self-Correcting Bayesian Optimization through Bayesian Active Learning

**Authors:** Carl Hvarfner, Erik Hellsten, Frank Hutter, Luigi Nardi

### Semi-Implicit Denoising Diffusion Models (SIDDMs)

**Authors:** yanwu xu, Mingming Gong, Shaoan Xie, Wei Wei, Matthias Grundmann, Kayhan Batmanghelich, Tingbo Hou

### [Spotlight] Semi-Supervised Domain Generalization with Known and Unknown Classes

**Authors:** Lei Zhang, Ji-Fu Li, Wei Wang

### Sequential Memory with Temporal Predictive Coding

**Authors:** Mufeng Tang, Helen Barron, Rafal Bogacz

### [Spotlight] Sharp Spectral Rates for Koopman Operator Learning

**Authors:** Vladimir Kostic, Karim Lounici, Pietro Novelli, Massimiliano Pontil

### [Oral] Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization

**Authors:** Kaiyue Wen, Zhiyuan Li, Tengyu Ma

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1D

### SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models

**Authors:** Hongxin Li, Jingran Su, Yuntao Chen, Qing Li, ZHAO-XIANG ZHANG

### Should We Learn Most Likely Functions or Parameters?

**Authors:** Shikai Qiu, Tim G. J. Rudner, Sanyam Kapoor, Andrew Wilson

### Similarity, Compression and Local Steps: Three Pillars of Efficient Communications for Distributed Variational Inequalities

**Authors:** Aleksandr Beznosikov, Martin Takac, Alexander Gasnikov

### SituatedGen: Incorporating Geographical and Temporal Contexts into Generative Commonsense Reasoning

**Authors:** Yunxiang Zhang, Xiaojun Wan

### Slimmed Asymmetrical Contrastive Learning and Cross Distillation for Lightweight Model Training

**Authors:** Jian Meng, Li Yang, Kyungmin Lee, Jinwoo Shin, Deliang Fan, Jae-sun Seo

### [Spotlight] Slow and Weak Attractor Computation Embedded in Fast and Strong E-I Balanced Neural Dynamics

**Authors:** Xiaohan Lin, Liyuan Li, Boxin Shi, Tiejun Huang, Yuanyuan Mi, Si Wu

### Small Transformers Compute Universal Metric Embeddings

**Authors:** Anastasis Kratsios, Valentin Debarnot, Ivan Dokmanić

### Smooth, exact rotational symmetrization for deep learning on point clouds

**Authors:** Sergey Pozdnyakov, Michele Ceriotti

### SmoothHess: ReLU Network Feature Interactions via Stein's Lemma

**Authors:** Max Torop, Aria Masoomi, Davin Hill, Kivanc Kose, Stratis Ioannidis, Jennifer Dy

### [Spotlight] Smoothed Analysis of Sequential Probability Assignment

**Authors:** Alankrita Bhatt, Nika Haghtalab, Abhishek Shetty

### SoTTA: Robust Test-Time Adaptation on Noisy Data Streams

**Authors:** Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, Sung-Ju Lee

### SoundCam: A Dataset for Finding Humans Using Room Acoustics

**Authors:** Mason Wang, Samuel Clarke, Jui-Hsien Wang, Ruohan Gao, Jiajun Wu

### Species196: A One-Million Semi-supervised Dataset for Fine-grained Species Recognition

**Authors:** Wei He, Kai Han, Ying Nie, Chengcheng Wang, Yunhe Wang

### SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents

**Authors:** Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang, Yongbin Li

### Stability Guarantees for Feature Attributions with Multiplicative Smoothing

**Authors:** Anton Xue, Rajeev Alur, Eric Wong

### Stabilized Neural Differential Equations for Learning Dynamics with Explicit Constraints

**Authors:** Alistair White, Niki Kilbertus, Maximilian Gelbrecht, Niklas Boers

### Stabilizing the Optimization of Neural Signed Distance Functions and Finer Shape Representation

**Authors:** Huizong Yang, Yuxin Sun, Ganesh Sundaramoorthi, Anthony Yezzi

### Stable Bias: Evaluating Societal Representations in Diffusion Models

**Authors:** Sasha Luccioni, Christopher Akiki, Margaret Mitchell, Yacine Jernite

### [Spotlight] Stable Nonconvex-Nonconcave Training via Linear Interpolation

**Authors:** Thomas Pethick, Wanyun Xie, Volkan Cevher

### State2Explanation: Concept-Based Explanations to Benefit Agent Learning and User Understanding

**Authors:** Devleena Das, Sonia Chernova, Been Kim

### StateMask: Explaining Deep Reinforcement Learning through State Mask

**Authors:** Zelei Cheng, Xian Wu, Jiahao Yu, Wenhai Sun, Wenbo Guo, Xinyu Xing

### [Spotlight] Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory

**Authors:** Sokhna Diarra Mbacke, Florence Clerc, Pascal Germain

### [Spotlight] Stochastic Multi-armed Bandits: Optimal Trade-off among Optimality, Consistency, and Tail Risk

**Authors:** David Simchi-Levi, Zeyu Zheng, Feng Zhu

### StoryBench: A Multifaceted Benchmark for Continuous Story Visualization

**Authors:** Emanuele Bugliarello, H. Hernan Moraldo, Ruben Villegas, Mohammad Babaeizadeh, Mohammad Taghi Saffar, Han Zhang, Dumitru Erhan, Vittorio Ferrari, Pieter-Jan Kindermans, Paul Voigtlaender

### StressID: a Multimodal Dataset for Stress Identification

**Authors:** Hava Chaptoukaev, Valeriya Strizhkova, Michele Panariello, Bianca Dalpaos, Aglind Reka, Valeria Manera, Susanne Thümmler, Esma ISMAILOVA, Nicholas W., francois bremond, Massimiliano Todisco, Maria A Zuluaga, Laura M. Ferrari

### Structure Learning with Adaptive Random Neighborhood Informed MCMC

**Authors:** Xitong Liang, Alberto Caron, Samuel Livingstone, Jim Griffin

### Subject-driven Text-to-Image Generation via Apprenticeship Learning

**Authors:** wenhu chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, William Cohen

### SubseasonalClimateUSA: A Dataset for Subseasonal Forecasting and Benchmarking

**Authors:** Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, Lester Mackey

### [Spotlight] Subspace Identification for Multi-Source Domain Adaptation

**Authors:** Zijian Li, Ruichu Cai, Guangyi Chen, Boyang Sun, Zhifeng Hao, Kun Zhang

### [Spotlight] Supervised Pretraining Can Learn In-Context Reinforcement Learning

**Authors:** Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill

### Supply-Side Equilibria in Recommender Systems

**Authors:** Meena Jagadeesan, Nikhil Garg, Jacob Steinhardt

### SustainGym: Reinforcement Learning Environments for Sustainable Energy Systems

**Authors:** Christopher Yeh, Victor Li, Rajeev Datta, Julio Arroyo, Nicolas Christianson, Chi Zhang, Yize Chen, Mohammad Mehdi Hosseini, Azarang Golmohammadi, Yuanyuan Shi, Yisong Yue, Adam Wierman

### Swap Agnostic Learning, or Characterizing Omniprediction via Multicalibration

**Authors:** Parikshit Gopalan, Michael Kim, Omer Reingold

### SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models

**Authors:** XIAOSONG MA, Jie ZHANG, Song Guo, Wenchao Xu

### SwiFT: Swin 4D fMRI Transformer

**Authors:** Peter Kim, Junbeom Kwon, Sunghwan Joo, Sangyoon Bae, Donggyu Lee, Yoonho Jung, Shinjae Yoo, Jiook Cha, Taesup Moon

### SynMob: Creating High-Fidelity Synthetic GPS Trajectory Dataset for Urban Mobility Analysis

**Authors:** Yuanshao Zhu, Yongchao Ye, Ying Wu, Xiangyu Zhao, James Yu

### Synthcity: a benchmark framework for diverse use cases of tabular synthetic data

**Authors:** Zhaozhi Qian, Rob Davis, Mihaela van der Schaar

### T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation

**Authors:** Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, Xihui Liu

### TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph

**Authors:** Xueyuan Lin, Haihong E, Chengjin Xu, Gengxian Zhou, Haoran Luo, Tianyi Hu, Fenglong Su, Ningyuan Li, Mingzhi Sun

### TRIAGE: Characterizing and auditing training data for improved regression

**Authors:** Nabeel Seedat, Jonathan Crabbé, Zhaozhi Qian, Mihaela van der Schaar

### Taking the neural sampling code very seriously: A data-driven approach for evaluating generative models of the visual system

**Authors:** Suhas Shrinivasan, Konstantin-Klemens Lurz, Kelli Restivo, George Denfield, Andreas Tolias, Edgar Walker, Fabian Sinz

### Taming Local Effects in Graph-based Spatiotemporal Forecasting

**Authors:** Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi

### Task-aware Distributed Source Coding under Dynamic Bandwidth

**Authors:** Po-han Li, Sravan Kumar Ankireddy, Ruihan (Philip) Zhao, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Ufuk Topcu, Sandeep Chinchali, Hyeji Kim

### TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery

**Authors:** Jialin Chen, Rex Ying

### Temporal Dynamic Quantization for Diffusion Models

**Authors:** Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, Eunhyeok Park

### Temporal Graph Benchmark for Machine Learning on Temporal Graphs

**Authors:** Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, Reihaneh Rabbany

### The Benefits of Being Distributional: Small-Loss Bounds for Reinforcement Learning

**Authors:** Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, Wen Sun

### The Best of Both Worlds in Network Population Games: Reaching Consensus and Convergence to Equilibrium

**Authors:** Shuyue Hu, Harold Soh, Georgios Piliouras

### The Cambridge Law Corpus: A Corpus for Legal AI Research

**Authors:** Andreas Östling, Holli Sargeant, Huiyuan Xie, Ludwig Bull, Alexander Terenin, Leif Jonsson, Måns Magnusson, Felix Steffek

### The Crucial Role of Normalization in Sharpness-Aware Minimization

**Authors:** Yan Dai, Kwangjun Ahn, Suvrit Sra

### The Double-Edged Sword of Implicit Bias: Generalization vs. Robustness in ReLU Networks

**Authors:** Spencer Frei, Gal Vardi, Peter Bartlett, Nati Srebro

### [Spotlight] The Equivalence of Dynamic and Strategic Stability under Regularized Learning in Games

**Authors:** Victor Boone, Panayotis Mertikopoulos

### The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications

**Authors:** Mirac Suzgun, Luke Melas-Kyriazi, Suproteem Sarkar, Scott D Kominers, Stuart Shieber

### The Target-Charging Technique for Privacy Analysis across Interactive Computations

**Authors:** Edith Cohen, Xin Lyu

### The probability flow ODE is provably fast

**Authors:** Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, Adil Salim

### Theoretical Analysis of the Inductive Biases in Deep Convolutional Networks

**Authors:** Zihao Wang, Lei Wu

### Thrust: Adaptively Propels Large Language Models with External Knowledge

**Authors:** Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Jianshu Chen

### [Spotlight] Tight Risk Bounds for Gradient Descent on Separable Data

**Authors:** Matan Schliserman, Tomer Koren

### To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning

**Authors:** Ildus Sadrtdinov, Dmitrii Pozdeev, Dmitry Vetrov, Ekaterina Lobacheva

### ToolQA: A Dataset for LLM Question Answering with External Tools

**Authors:** Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, Chao Zhang

### TopoSRL: Topology preserving self-supervised Simplicial Representation Learning

**Authors:** Hiren Madhu, Sundeep Prabhakar Chepuri

### Topological RANSAC for instance verification and retrieval without fine-tuning

**Authors:** Guoyuan An, Ju-hyeong Seon, Inkyu An, Yuchi Huo, Sung-eui Yoon

### Topology-Aware Uncertainty for Image Segmentation

**Authors:** Saumya Gupta, Yikai Zhang, Xiaoling Hu, Prateek Prasanna, Chao Chen

### Towards Efficient Pre-Trained Language Model via Feature Correlation Distillation

**Authors:** Kun Huang, Xin Guo, Meng Wang

### Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior

**Authors:** Shashank Subramanian, Peter Harrington, Kurt Keutzer, Wahid Bhimji, Dmitriy Morozov, Michael Mahoney, Amir Gholami

### [Spotlight] Towards In-context Scene Understanding

**Authors:** Ivana Balazevic, David Steiner, Nikhil Parthasarathy, Relja Arandjelović, Olivier Henaff

### Towards Last-Layer Retraining for Group Robustness with Fewer Annotations

**Authors:** Tyler LaBonte, Vidya Muthukumar, Abhishek Kumar

### Towards Last-layer Retraining for Group Robustness with Fewer Annotations

**Authors:** Tyler LaBonte, Vidya Muthukumar, Abhishek Kumar

### Towards Unbounded Machine Unlearning

**Authors:** Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou

### Towards a Unified Analysis of Kernel-based Methods Under Covariate Shift

**Authors:** Xingdong Feng, Xin HE, Caixing Wang, Chao Wang, Jingnan Zhang

### Towards a fuller understanding of neurons with Clustered Compositional Explanations

**Authors:** Biagio La Rosa, Leilani Gilpin, Roberto Capobianco

### TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

**Authors:** Mangpo Phothilimtha, Sami Abu-El-Haija, Kaidi Cao, Bahare Fatemi, Michael Burrows, Charith Mendis, Bryan Perozzi

### [Spotlight] Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning

**Authors:** Shenzhi Wang, Qisen Yang, Jiawei Gao, Matthieu Lin, HAO CHEN, Liwei Wu, Ning Jia, Shiji Song, Gao Huang

### Training Chain-of-Thought via Latent-Variable Inference

**Authors:** Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous

### [Spotlight] Training shallow ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?

**Authors:** Erin George, Michael Murray, William Swartworth, Deanna Needell

### Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis

**Authors:** Zhiyu Jin, Xuli Shen, Bin Li, Xiangyang Xue

### Transformers learn through gradual rank increase

**Authors:** Emmanuel Abbe, Samy Bengio, Enric Boix-Adsera, Etai Littwin, Joshua Susskind

### Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images

**Authors:** Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein

### Triangulation Residual Loss for Data-efficient 3D Pose Estimation

**Authors:** Jiachen Zhao, Tao Yu, Liang An, Yipeng Huang, Fang Deng, Qionghai Dai

### Turbulence in Focus: Benchmarking Scaling Behavior of 3D Volumetric Super-Resolution with BLASTNet 2.0 Data

**Authors:** Wai Tong Chung, Bassem Akoush, Pushan Sharma, Alex Tamkin, Ki Sung Jung, Jacqueline Chen, Jack Guo, Davy Brouzet, Mohsen Talei, Bruno Savard, Alexei Poludnenko, Matthias Ihme

### UUKG: Unified Urban Knowledge Graph Dataset for Urban Spatiotemporal Prediction

**Authors:** Yansong Ning, Hao Liu, Hao Wang, Zhenyu Zeng, Hui Xiong

### [Spotlight] Uncovering motifs of concurrent signaling across multiple neuronal populations

**Authors:** Evren Gokcen, Anna Jasper, Alison Xu, Adam Kohn, Christian Machens, Byron M Yu

### Understanding and Mitigating Copying in Diffusion Models

**Authors:** Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein

### Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

**Authors:** Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, Kwan-Yee K. Wong

### UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

**Authors:** Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu

### UniT: A Unified Look at Certified Robust Training against Text Adversarial Perturbation

**Authors:** Muchao Ye, Ziyi Yin, Tianrong Zhang, Tianyu Du, Jinghui Chen, Ting Wang, Fenglong Ma

### Unified Lower Bounds for Interactive High-dimensional Estimation under Information Constraints

**Authors:** Jayadev Acharya, Clément L Canonne, Ziteng Sun, Himanshu Tyagi

### Unified Segment-to-Segment Framework for Simultaneous Sequence Generation

**Authors:** Shaolei Zhang, Yang Feng

### Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction

**Authors:** Qing Wu, Lixuan Chen, Ce Wang, Hongjiang Wei, S. Kevin Zhou, Jingyi Yu, Yuyao Zhang

### VPP: Efficient Conditional 3D Generation via Voxel-Point Progressive Representation

**Authors:** Zekun Qi, Muzhou Yu, Runpei Dong, Kaisheng Ma

### VTaC: A  Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors

**Authors:** Li-wei Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, Roger Mark

### Validated Image Caption Rating Dataset

**Authors:** Lothar D Narins, Andrew Scott, Aakash Gautam, Anagha Kulkarni, Mar Castanon, Benjamin Kao, Shasta Ihorn, Yue-Ting Siu, James M. Mason, Alexander Blum, Ilmi Yoon

### Variational Imbalanced Regression: Fair Uncertainty Quantification via Probabilistic Smoothing

**Authors:** Ziyan Wang, Hao Wang

### Video-Mined Task Graphs for Keystep Recognition in Instructional Videos

**Authors:** Kumar Ashutosh, Santhosh Kumar Ramakrishnan, Triantafyllos Afouras, Kristen Grauman

### VisIT-Bench: A Dynamic Benchmark for Evaluating Instruction-Following Vision-and-Language Models

**Authors:** Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, Ludwig Schmidt

### WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding

**Authors:** Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, Carl Yang

### Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research

**Authors:** Cole Gulino, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, Xinlei Pan, Yan Wang, Xiangyu Chen, John Co-Reyes, Rishabh Agarwal, Rebecca Roelofs, Yao Lu, Nico Montali, Paul Mougin, Zoey Yang, Brandyn White, Aleksandra Faust, Rowan McAllister, Dragomir Anguelov, Benjamin Sapp

### Weitzman's Rule for Pandora's Box with Correlations

**Authors:** Evangelia Gergatsouli, Christos Tzamos

### What Distributions are Robust to Indiscriminate Poisoning Attacks for Linear Learners?

**Authors:** Fnu Suya, Xiao Zhang, Yuan Tian, David Evans

### What Truly Matters in Trajectory Prediction for Autonomous Driving?

**Authors:** Tran Phong, Haoran Wu, Cunjun Yu, Panpan Cai, Sifa Zheng, David Hsu

### What You See is What You Read? Improving Text-Image Alignment Evaluation

**Authors:** Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, Idan Szpektor

### What a MESS: Multi-Domain Evaluation of Zero-Shot Semantic Segmentation

**Authors:** Benedikt Blumenstiel, Johannes Jakubik, Hilde Kuehne, Michael Vössing

### What’s Left? Concept Grounding with Logic-Enhanced Foundation Models

**Authors:** Joy Hsu, Jiayuan Mao, Josh Tenenbaum, Jiajun Wu

### [Oral] When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning

**Authors:** Siliang Zeng, Chenliang Li, Alfredo Garcia, Mingyi Hong

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1A

### When Does Confidence-Based Cascade Deferral Suffice?

**Authors:** Wittawat Jitkrittum, Neha Gupta, Aditya Menon, Harikrishna Narasimhan, Ankit Rawat, Sanjiv Kumar

### When is Agnostic Reinforcement Learning Statistically Tractable?

**Authors:** Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, Nati Srebro

### Where Did I Come From? Origin Attribution of AI-Generated Images

**Authors:** Zhenting Wang, Chen Chen, Yi Zeng, Lingjuan Lyu, Shiqing Ma

### Why Does Sharpness-Aware Minimization Generalize Better Than SGD?

**Authors:** Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, Quanquan Gu

### WordScape: a Pipeline to extract multilingual, visually rich Documents with Layout Annotations from Web Crawl Data

**Authors:** Maurice Weber, Carlo Siebenschuh, Rory Butler, Anton Alexandrov, Valdemar Thanner, Georgios Tsolakis, Haris Jabbar, Ian Foster, Bo Li, Rick Stevens, Ce Zhang

### Wyze Rule: Federated Rule Dataset for Rule Recommendation Benchmarking

**Authors:** Mohammad Mahdi Kamani, Yuhang Yao, Hanjia Lyu, Zhongwei Cheng, Lin Chen, Liangju Li, Carlee Joe-Wong, Jiebo Luo

### XAGen: 3D Expressive Human Avatars Generation

**Authors:** Zhongcong XU, Jianfeng Zhang, Jun Hao Liew, Jiashi Feng, Mike Zheng Shou

### Zero-Shot Anomaly Detection via Batch Normalization

**Authors:** Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, Stephan Mandt

### Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models

**Authors:** Lin Li, Jun Xiao, Guikun Chen, Jian Shao, Yueting Zhuang, Long Chen

### Zeroth-Order Methods for Nondifferentiable, Nonconvex, and Hierarchical Federated Optimization

**Authors:** Yuyang Qiu, Uday Shanbhag, Farzad Yousefian

### Zeroth-Order Methods for Nonsmooth, Nonconvex, and Hierarchical Federated Optimization

**Authors:** Yuyang Qiu, Uday Shanbhag, Farzad Yousefian

### [Re] Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods

**Authors:** Domen Mohorčič, David Ocepek

### [Re] Masked Autoencoders Are Small Scale Vision Learners: A Reproduction Under Resource Constraints

**Authors:** Athanasios Charisoudis, Simon Ekman von Huth, Emil Jansson

### [Re] VAE Approximation Error: ELBO and Exponential Families

**Authors:** Volodymyr Kyrylov, Navdeep Singh Bedi, Qianbo Zang

### rPPG-Toolbox: Deep Remote PPG Toolbox

**Authors:** Xin Liu, Girish Narayanswamy, Akshay Paruchuri, Xiaoyu Zhang, Jiankai Tang, Yuzhe Zhang, Roni Sengupta, Shwetak Patel, Yuntao Wang, Daniel McDuff

</details>

<details><summary><h3 style='display: inline;'> Poster Session 2: Tuesday, Dec 12, 15:15 CT</h3></summary>

### (Almost) Provable Error Bounds Under Distribution Shift via Disagreement Discrepancy

**Authors:** Elan Rosenfeld, Saurabh Garg

### 3D-Aware Visual Question Answering about Parts, Poses and Occlusions

**Authors:** Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, Alan Yuille

### A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings

**Authors:** Nathaniel Lahn, Sharath Raghvendra, Kaiyi Zhang

### A Competitive Algorithm for Agnostic Active Learning

**Authors:** Yihan Zhou, Eric Price

### A Computationally Efficient Sparsified Online Newton Method

**Authors:** Fnu Devvrit, Sai Surya Duvvuri, Rohan Anil, Vineet Gupta, Cho-Jui Hsieh, Inderjit Dhillon

### [Spotlight] A Cross-Moment Approach for Causal Effect Estimation

**Authors:** Yaroslav Kivva, Saber Salehkaleybar, Negar Kiyavash

### A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks

**Authors:** Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, Salman Avestimehr

### [Spotlight] A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability

**Authors:** Zijie Geng, Xijun Li, Jie Wang, Xiao Li, Yongdong Zhang, Feng Wu

### A Diffusion-Model of Joint Interactive Navigation

**Authors:** Matthew Niedoba, Jonathan Lavington, Yunpeng Liu, Vasileios Lioutas, Justice Sefas, Xiaoxuan Liang, Dylan Green, Setareh Dabiri, Berend Zwartsenberg, Adam Scibior, Frank Wood

### A Finite-Particle Convergence Rate for Stein Variational Gradient Descent

**Authors:** Jiaxin Shi, Lester Mackey

### A General Framework for Equivariant Neural Networks on Reductive Lie Groups

**Authors:** Ilyes Batatia, Mario Geiger, Jose Munoz, Tess Smidt, Lior Silberman, Christoph Ortner

### A General Theory of Correct, Incorrect, and Extrinsic Equivariance

**Authors:** Dian Wang, Xupeng Zhu, Jung Yeon Park, Mingxi Jia, Guanang Su, Robert Platt, Robin Walters

### [Oral] A Measure-Theoretic Axiomatisation of Causality

**Authors:** Junhyung Park, Simon Buchholz, Bernhard Schölkopf, Krikamol Muandet

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2C

### A Reduction-based Framework for Sequential Decision Making with Delayed Feedback

**Authors:** Yunchang Yang, Han Zhong, Tianhao Wu, Bin Liu, Liwei Wang, Simon Du

### A Riemannian Exponential Augmented Lagrangian Method for Computing the Projection Robust Wasserstein Distance

**Authors:** Bo Jiang, Ya-Feng Liu

### A Simple Solution for Offline Imitation from Observations and Examples with Possibly Incomplete Trajectories

**Authors:** Kai Yan, Alex Schwing, Yu-Xiong Wang

### [Spotlight] A Spectral Theory of Neural Prediction and Alignment

**Authors:** Abdulkadir Canatar, Jenelle Feather, Albert Wakhloo, SueYeon Chung

### A State Representation for Diminishing Rewards

**Authors:** Ted Moskovitz, Samo Hromadka, Ahmed Touati, Diana Borsa, Maneesh Sahani

### A Sublinear-Time Spectral Clustering Oracle with Improved Preprocessing Time

**Authors:** Ranran Shen, Pan Peng

### A Theoretical Analysis of Optimistic Proximal Policy Optimization in Linear Markov Decision Processes

**Authors:** Han Zhong, Tong Zhang

### A Theory of Multimodal Learning

**Authors:** Zhou Lu

### A Theory of Transfer-Based Black-Box Attacks: Explanation and Implications

**Authors:** Yanbo Chen, Weiwei Liu

### A Unified Detection Framework for Inference-Stage Backdoor Defenses

**Authors:** Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, Jie Ding

### A Unified Framework for U-Net Design and Analysis

**Authors:** Christopher Williams, Fabian Falck, George Deligiannidis, Chris C Holmes, Arnaud Doucet, Saifuddin Syed

### A Unified Solution for Privacy and Communication Efficiency in Vertical Federated Learning

**Authors:** Ganyu Wang, Bin Gu, Qingsong Zhang, Xiang Li, Boyu Wang, Charles Ling

### A normative theory of social conflict

**Authors:** Sergey Shuvaev, Evgeny Amelchenko, Dmitry Smagin, Natalia Kudryavtseva, Grigori Enikolopov, Alex Koulakov

### AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix

**Authors:** Yun Yue, Zhiling Ye, Jiadi Jiang, Yongchao Liu, Ke Zhang

### AI for Interpretable Chemistry: Predicting Radical Mechanistic Pathways via Contrastive Learning

**Authors:** Mohammadamin Tavakoli, Pierre Baldi, Ann Marie Carlton, Yin Ting Chiu, Alexander Shmakov, David Van Vranken

### AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neuron Activity

**Authors:** Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, Eli Shlizerman

### AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation

**Authors:** Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, yelong shen, Jian Jiao, Juntao Li, zhongyu wei, Jian Guo, Nan Duan, Weizhu Chen

### ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition

**Authors:** Aashaka Desai, Lauren Berger, Fyodor Minakov, Nessa Milano, Chinmay Singh, Kriston Pumphrey, Richard Ladner, Hal Daumé III, Alex X Lu, Naomi Caselli, Danielle Bragg

### AVIS: Autonomous Visual Information Seeking with Large Language Model Agent

**Authors:** Ziniu Hu, Ahmet Iscen, Chen Sun, Kai-Wei Chang, Yizhou Sun, David Ross, Cordelia Schmid, Alireza Fathi

### Accelerated On-Device Forward Neural Network Training with Module-Wise Descending Asynchronism

**Authors:** Xiaohan Zhao, Hualin Zhang, Zhouyuan Huo, Bin Gu

### Accelerating Molecular Graph Neural Networks via Knowledge Distillation

**Authors:** Filip Ekström Kelvinius, Dimitar Georgiev, Artur Toshev, Johannes Gasteiger

### Accelerating Monte Carlo Tree Search with Probability Tree State Abstraction

**Authors:** Yangqing Fu, Ming Sun, Buqing Nie, Yue Gao

### Accelerating Motion Planning via Optimal Transport

**Authors:** An T. Le, Georgia Chalvatzaki, Armin Biess, Jan Peters

### Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration

**Authors:** Dongyoung Kim, Jinwoo Shin, Pieter Abbeel, Younggyo Seo

### Accurate Interpolation for Scattered Data through Hierarchical Residual Refinement

**Authors:** Shizhe Ding, Boyang Xia, Dongbo Bu

### Active Bipartite Ranking

**Authors:** James Cheshire, Vincent Laurent, Stephan Clémençon

### Adapting Fairness Interventions to Missing Values

**Authors:** Raymond Feng, Flavio Calmon, Hao Wang

### Adapting Neural Link Predictors for Data-Efficient Complex Query Answering

**Authors:** Erik Arakelyan, Pasquale Minervini, Daniel Daza, Michael Cochez, Isabelle Augenstein

### Adaptive Contextual Perception: How To Generalize To New Backgrounds and Ambiguous Objects

**Authors:** Zhuofan Ying, Peter Hase, Mohit Bansal

### Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective

**Authors:** Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya Huang, Qi Liu, Yanhu Xie, Enhong Chen

### Adaptive Selective Sampling for Online Prediction with Experts

**Authors:** Rui Castro, Fredrik Hellström, Tim van Erven

### Adaptive Topological Feature via Persistent Homology: Filtration Learning for Point Clouds

**Authors:** Naoki Nishikawa, Yuichi Ike, Kenji Yamanishi

### Adaptive Uncertainty Estimation via High-Dimensional Testing on Latent Representations

**Authors:** Tsai Hor Chan, Kin Wai Lau, Jiajun Shen, Guosheng Yin, Lequan Yu

### [Oral] Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation

**Authors:** Sébastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, Simon Lacoste-Julien

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2B

### Addressing Negative Transfer in Diffusion Models

**Authors:** Hyojun Go, Kim, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, Seungtaek Choi

### Adjustable Robust Reinforcement Learning for Online 3D Bin Packing

**Authors:** Yuxin Pan, Yize Chen, Fangzhen Lin

### Adversarial Model for Offline Reinforcement Learning

**Authors:** Mohak Bhardwaj, Tengyang Xie, Byron Boots, Nan Jiang, Ching-An Cheng

### Adversarially Robust Distributed Count Tracking via Partial Differential Privacy

**Authors:** Zhongzheng Xiong, Xiaoyi Zhu, zengfeng Huang

### Adversarially Robust Learning with Uncertain Perturbation Sets

**Authors:** Tosca Lechner, Vinayak Pathak, Ruth Urner

### Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization

**Authors:** Jameel Abdul Samadh, Mohammad Hanan Gani, Noor Hussein, Muhammad Uzair Khattak, Muhammad Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

### [Spotlight] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback

**Authors:** Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, Tatsunori Hashimoto

### AmadeusGPT: a natural language interface for interactive animal behavioral analysis

**Authors:** shaokai ye, Jessy Lauer, Mu Zhou, Alexander Mathis, Mackenzie Mathis

### An Alternative to Variance: Gini Deviation for Risk-averse Policy Gradient

**Authors:** Yudong Luo, Guiliang Liu, Pascal Poupart, Yangchen Pan

### An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination

**Authors:** Xue Yan, Jiaxian Guo, Xingzhou Lou, Jun Wang, Haifeng Zhang, Yali Du

### An Improved Relaxation for Oracle-Efficient Adversarial Contextual Bandits

**Authors:** Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Max Springer

### Analyzing Generalization of Neural Networks through Loss Path Kernels

**Authors:** Yilan Chen, Wei Huang, Hao Wang, Charlotte Loh, Akash Srivastava, Lam Nguyen, Lily Weng

### Any-to-Any Generation via Composable Diffusion

**Authors:** Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal

### Attention as Implicit Structural Inference

**Authors:** Ryan Singh, Christopher L Buckley

### Augmenting Language Models with Long-Term Memory

**Authors:** Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, Furu Wei

### Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning

**Authors:** Yifan Zang, Jinmin He, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng

### BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization

**Authors:** Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, Jean-Marc Andreoli

### Balancing Risk and Reward: A Batched-Bandit Strategy for Automated Phased Release

**Authors:** Yufan Li, Jialiang Mao, Iavor Bojinov

### [Spotlight] Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance

**Authors:** Congyue Deng, Jiahui Lei, William B Shen, Kostas Daniilidis, Leonidas Guibas

### Bayesian Learning via Q-Exponential Process

**Authors:** Shuyi Li, Michael O'Connor, Shiwei Lan

### Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval

**Authors:** Frederik Warburg, Marco Miani, Silas Brack, Søren Hauberg

### Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis

**Authors:** Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete

### Beyond Invariance: Test-Time Label-Shift Adaptation for Addressing "Spurious" Correlations

**Authors:** Qingyao Sun, Kevin Murphy, Sayna Ebrahimi, Alexander D'Amour

### Beyond Normal: On the Evaluation of Mutual Information Estimators

**Authors:** Paweł Czyż, Frederic Grabowski, Julia Vogt, Niko Beerenwinkel, Alexander Marx

### Beyond probability partitions: Calibrating neural networks with semantic aware grouping

**Authors:** Jia-Qi Yang, De-Chuan Zhan, Le Gan

### Bias in Evaluation Processes: An Optimization-Based Model

**Authors:** L. Elisa Celis, Amit Kumar, Anay Mehrotra, Nisheeth K. Vishnoi

### Bicriteria Multidimensional Mechanism Design with Side Information

**Authors:** Siddharth Prasad, Maria-Florina Balcan, Tuomas Sandholm

### Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm

**Authors:** Jie Hao, Kaiyi Ji, Mingrui Liu

### [Spotlight] Birth of a Transformer: A Memory Viewpoint

**Authors:** Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, Leon Bottou

### Block-Coordinate Methods and Restarting for Solving Extensive-Form Games

**Authors:** Darshan Chakrabarti, Jelena Diakonikolas, Christian Kroer

### Boosting Adversarial Transferability by Achieving Flat Local Maxima

**Authors:** Zhijin Ge, Wang Xiaosen, Hongying Liu, Fanhua Shang, Yuanyuan Liu

### [Spotlight] Bootstrapping Vision-Language Learning with Decoupled Language Pre-training

**Authors:** Yiren Jian, Chongyang Gao, Soroush Vosoughi

### Bounding training data reconstruction in DP-SGD

**Authors:** Jamie Hayes, Borja Balle, Saeed Mahloujifar

### Breaking the Communication-Privacy-Accuracy Tradeoff with $f$-Differential Privacy

**Authors:** Richeng Jin, Zhonggen Su, caijun zhong, Zhaoyang Zhang, Tony Quek, Huaiyu Dai

### [Oral] Bridging Discrete and Backpropagation: Straight-Through and Beyond

**Authors:** Liyuan Liu, Chengyu Dong, Xiaodong Liu, Bin Yu, Jianfeng Gao

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2A

### Budgeting Counterfactual for Offline RL

**Authors:** Yao Liu, Pratik Chaudhari, Rasool Fakoor

### CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society

**Authors:** Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem

### CAPro: Webly Supervised Learning with Cross-modality Aligned Prototypes

**Authors:** Yulei Qin, Xingyu Chen, Yunhang Shen, Chaoyou Fu, Yun Gu, Ke Li, Xing Sun, Rongrong Ji

### CRoSS: Diffusion Model Makes Controllable, Robust and Secure Image Steganography

**Authors:** Jiwen Yu, Xuanyu Zhang, Youmin Xu, Jian Zhang

### Cal-DETR: Calibrated Detection Transformer

**Authors:** Muhammad Akhtar Munir, Salman Khan, Muhammad Haris Khan, Mohsen Ali, Fahad Shahbaz Khan

### Calibrate and Boost Logical Expressiveness of GNN Over Multi-Relational and Temporal Graphs

**Authors:** Yeyuan Chen, Dingmin Wang

### CamoPatch: An Evolutionary Strategy for Generating Camoflauged Adversarial Patches

**Authors:** Phoenix Williams, Ke Li

### Cascading Contextual Assortment Bandits

**Authors:** Hyun-jun Choi, Rajan Udwani, Min-hwan Oh

### Causal Fairness for Outcome Control

**Authors:** Drago Plecko, Elias Bareinboim

### Causal discovery from observational and interventional data across multiple environments

**Authors:** Adam Li, Amin Jaber, Elias Bareinboim

### [Oral] Causal normalizing flows: from theory to practice

**Authors:** Adrián Javaloy, Pablo Sanchez-Martin, Isabel Valera

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2C

### Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks

**Authors:** Alex Schlögl, Nora Hofer, Rainer Böhme

### Certified Minimax Unlearning with Generalization Rates and Deletion Capacity

**Authors:** Jiaqi Liu, Jian Lou, Zhan Qin, Kui Ren

### Change point detection and inference in multivariate non-parametric models under mixing conditions

**Authors:** Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu

### [Spotlight] Characterizing the Optimal $0-1$ Loss for Multi-class Classification with a Test-time Attacker

**Authors:** Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Heather Zheng, Ben Zhao, Prateek Mittal

### Closing the Computational-Statistical Gap in Best Arm Identification for Combinatorial Semi-bandits

**Authors:** Ruo-Chun Tzeng, Po-An Wang, Alexandre Proutiere, Chi-Jen Lu

### Clustering the Sketch: Dynamic Compression for Embedding Tables

**Authors:** Henry Tsang, Thomas Ahle

### CoLLAT: On Adding Fine-grained Audio Understanding to Language Models using Token-Level Locked-Language Tuning

**Authors:** Dadallage A R Silva, Spencer Whitehead, Christopher Lengerich, Hugh Leather

### Collaborative Score Distillation for Consistent Visual Editing

**Authors:** Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, Jinwoo Shin

### ComSL: A Composite Speech-Language Model for End-to-End Speech-to-Text Translation

**Authors:** Chenyang Le, Yao Qian, Long Zhou, Shujie LIU, Yanmin Qian, Michael Zeng, Xuedong Huang

### Combinatorial Group Testing with Selfish Agents

**Authors:** Georgios Chionas, Dariusz Kowalski, Piotr Krysta

### [Spotlight] Common Ground in Cooperative Communication

**Authors:** Xiaoran Hao, Yash Jhaveri, Patrick Shafto

### CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs

**Authors:** Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, Benjamin Busam

### Communication-Efficient Federated Bilevel Optimization with Global and Local Lower Level Problems

**Authors:** Junyi Li, Feihu Huang, Heng Huang

### Comparing Causal Frameworks: Potential Outcomes, Structural Models, Graphs, and Abstractions

**Authors:** Duligur Ibeling, Thomas Icard

### [Spotlight] Complexity Matters: Rethinking the Latent Space for Generative Modeling

**Authors:** Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, Zhenguo Li

### Composing Parameter-Efficient Modules with Arithmetic Operation

**Authors:** Jinghan Zhang, shiqi chen, Junteng Liu, Junxian He

### Compressed Video Prompt Tuning

**Authors:** Bing Li, Jiaxin Chen, Xiuguo Bao, Di Huang

### [Spotlight] Compression with Bayesian Implicit Neural Representations

**Authors:** Zongyu Guo, Gergely Flamich, Jiajun He, Zhibo Chen, José Miguel Hernández-Lobato

### Computational Complexity of Learning Neural Networks: Smoothness and Degeneracy

**Authors:** Amit Daniely, Nati Srebro, Gal Vardi

### [Spotlight] Computing a human-like reaction time metric from stable recurrent vision models

**Authors:** Lore Goetschalckx, Alekh Karkada Ashok, Aarit Ahuja, David Sheinberg, Thomas Serre

### Coneheads: Hierarchy Aware Attention

**Authors:** Albert Tseng, Tao Yu, Toni Liu, Christopher De Sa

### [Oral] Conformal Meta-learners for Predictive Inference of Individual Treatment Effects

**Authors:** Ahmed Alaa, Zaid Ahmad, Mark van der Laan

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2C

### Conformalized matrix completion

**Authors:** Yu Gui, Rina Barber, Cong Ma

### Conservative State Value Estimation for Offline Reinforcement Learning

**Authors:** Liting Chen, Jie Yan, Zhengdao Shao, Lu Wang, Qingwei Lin, Saravanakumar Rajmohan, Thomas Moscibroda, Dongmei Zhang

### [Spotlight] Constant Approximation for Individual Preference Stable Clustering

**Authors:** Anders Aamand, Justin Chen, Allen Liu, Sandeep Silwal, Pattara Sukprasert, Ali Vakilian, Fred Zhang

### [Spotlight] Convex and Non-convex Optimization Under Generalized Smoothness

**Authors:** Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, Ali Jadbabaie

### Convex-Concave Zero-Sum Stochastic Stackelberg Games

**Authors:** Denizalp Goktas, Arjun Prakash, Amy Greenwald

### Convolution Monge Mapping Normalization for learning on sleep data

**Authors:** Théo Gnassounou, Rémi Flamary, Alexandre Gramfort

### Convolutional State Space Models for Long-Range Spatiotemporal Modeling

**Authors:** Jimmy Smith, Shalini De Mello, Jan Kautz, Scott Linderman, Wonmin Byeon

### Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP

**Authors:** Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, Liang-Chieh Chen

### Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning

**Authors:** Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, Xiangyang Ji

### [Spotlight] Counterfactual Memorization in Neural Language Models

**Authors:** Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramer, Nicholas Carlini

### Counterfactually Comparing Abstaining Classifiers

**Authors:** Yo Joong Choe, Aditya Gangrade, Aaditya Ramdas

### Counting Distinct Elements in the Turnstile Model with Differential Privacy under Continual Observation

**Authors:** Palak Jain, Iden Kalemaj, Sofya Raskhodnikova, Satchit Sivakumar, Adam Smith

### Cross-Scale MAE: A Tale of Multiscale Exploitation in Remote Sensing

**Authors:** Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, Hairong Qi

### DAC-DETR: Divide the Attention Layers and Conquer

**Authors:** Zhengdong Hu, Yifan Sun, Jingdong Wang, Yi Yang

### DDF-HO: Hand-Held Object Reconstruction via Conditional Directed Distance Field

**Authors:** Chenyangguang Zhang, Yan Di, Ruida Zhang, Guangyao Zhai, Fabian Manhardt, Federico Tombari, Xiangyang Ji

### DELTA: Diverse Client Sampling for Fasting Federated Learning

**Authors:** Lin Wang, Yongxin Guo, Tao Lin, Xiaoying Tang

### DFRD: Data-Free Robustness Distillation for Heterogeneous Federated Learning

**Authors:** kangyang Luo, Shuai Wang, Yexuan Fu, Xiang Li, Yunshi Lan, Ming Gao

### DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement

**Authors:** Wenxin Tai, Yue Lei, Fan Zhou, Goce Trajcevski, Ting Zhong

### DP-HyPO: An Adaptive Private Framework for Hyperparameter Optimization

**Authors:** Hua Wang, Sheng Gao, Huanyu Zhang, Weijie Su, Milan Shen

### DP-Mix: Mixup-based Data Augmentation for Differentially Private Learning

**Authors:** Wenxuan Bao, Francesco Pittaluga, Vijay Kumar B G, Vincent Bindschaedler

### DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics

**Authors:** Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu

### Data-Dependent Bounds for Online Portfolio Selection Without Lipschitzness and Smoothness

**Authors:** Chung-En Tsai, Ying-Ting Lin, Yen-Huan Li

### Data-driven Optimal Filtering for Linear Systems with Unknown Noise Covariances

**Authors:** Shahriar Talebi, Amirhossein Taghvaei, Mehran Mesbahi

### De novo Drug Design using Reinforcement Learning with Multiple GPT Agents

**Authors:** Xiuyuan Hu, Guoqing Liu, Yang Zhao, Hao Zhang

### [Spotlight] Debias Coarsely, Sample Conditionally: Statistical Downscaling through Optimal Transport and Probabilistic Diffusion Models

**Authors:** Zhong Yi Wan, Ricardo Baptista, Anudhyan Boral, Yi-Fan Chen, John Anderson, Fei Sha, Leonardo Zepeda-Núñez

### Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment

**Authors:** Yutong Xia, Yuxuan Liang, Haomin Wen, Xu Liu, Kun Wang, Zhengyang Zhou, Roger Zimmermann

### Decision-Aware Actor-Critic with Function Approximation and Theoretical Guarantees

**Authors:** Sharan Vaswani, Amirreza Kazemi, Reza Babanezhad Harikandeh, Nicolas Le Roux

### Decorate3D: Text-Driven High-Quality Texture Generation for Mesh Decoration in the Wild

**Authors:** Yanhui Guo, Xinxin Zuo, Peng Dai, Juwei Lu, Xiaolin Wu, Li cheng, Youliang Yan, Songcen Xu, Xiaofei Wu

### Deep learning with kernels through RKHM and the Perron-Frobenius operator

**Authors:** Yuka Hashimoto, Masahiro Ikeda, Hachem Kadri

### Defending against Data-Free Model Extraction by  Distributionally Robust Defensive Training

**Authors:** Zhenyi Wang, Li Shen, Tongliang Liu, Tiehang Duan, Yanjun Zhu, Donglin Zhan, DAVID DOERMANN, Mingchen Gao

### [Spotlight] Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models

**Authors:** Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, Leonid Karlinsky

### Described Object Detection: Liberating Object Detection with Flexible Expressions

**Authors:** Chi Xie, Zhao Zhang, Yixuan Wu, Feng Zhu, Rui Zhao, Shuang Liang

### Designing Robust Transformers using Robust Kernel Density Estimation

**Authors:** Xing Han, Tongzheng Ren, Tan Nguyen, Khai Nguyen, Joydeep Ghosh, Nhat Ho

### Differentiable Random Partition Models

**Authors:** Thomas Sutter, Alain Ryser, Joram Liebeskind, Julia Vogt

### Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes

**Authors:** Ali Younis, Erik Sudderth

### Differentially Private Statistical Inference through $\beta$-Divergence One Posterior Sampling

**Authors:** Jack Jewson, Sahra Ghalebikesabi, Chris C Holmes

### Diffused Task-Agnostic Milestone Planner

**Authors:** Mineui Hong, Minjae Kang, Songhwai Oh

### Diffusion Model for Graph Inverse Problems: Towards Effective Source Localization on Complex Networks

**Authors:** Xin Yan, Hui Fang, Qiang He

### Diffusion Representation for Asymmetric Kernels via Magnetic Transform

**Authors:** Mingzhen He, FAN He, Ruikai Yang, Xiaolin Huang

### Direct Diffusion Bridge using Data Consistency for Inverse Problems

**Authors:** Hyungjin Chung, Jeongsol Kim, Jong Chul Ye

### Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms

**Authors:** Peiyao Xiao, Hao Ban, Kaiyi Ji

### DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models

**Authors:** Tao Yang, Yuwang Wang, Yan Lu, Nanning Zheng

### [Spotlight] Distance-Restricted Folklore Weisfeiler-Leman GNNs with Provable Cycle Counting Power

**Authors:** Junru Zhou, Jiarui Feng, Xiyuan Wang, Muhan Zhang

### Diversify Your Vision Datasets with Automatic Diffusion-based Augmentation

**Authors:** Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph Gonzalez, Trevor Darrell

### Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback

**Authors:** Jaskirat Singh, Liang Zheng

### Django: Detecting Trojans in Object Detection Models via Gaussian Focus Calibration

**Authors:** Guangyu Shen, Siyuan Cheng, Guanhong Tao, Kaiyuan Zhang, Yingqi Liu, Shengwei An, Shiqing Ma, Xiangyu Zhang

### Double Auctions with Two-sided Bandit Feedback

**Authors:** Soumya Basu, Abishek Sankararaman

### Dual Self-Awareness Value Decomposition Framework without Individual Global Max for Cooperative MARL

**Authors:** Zhiwei Xu, Bin Zhang, dapeng li, Guangchong Zhou, Zeren Zhang, Guoliang Fan

### Dynamic Non-monotone Submodular Maximization

**Authors:** Kiarash Banihashem, Leyla Biabani, Samira Goudarzi, MohammadTaghi Hajiaghayi, Peyman Jabbarzade, Morteza Monemizadeh

### Dynamic Prompt Learning: Addressing Cross-Attention Leakage for Text-Based Image Editing

**Authors:** kai wang, Fei Yang, Shiqi Yang, Muhammad Atif Butt, Joost van de Weijer

### Dynamic Regret of Adversarial Linear Mixture MDPs

**Authors:** Long-Fei Li, Peng Zhao, Zhi-Hua Zhou

### Dynamic Sparsity Is Channel-Level Sparsity Learner

**Authors:** Lu Yin, Gen Li, Meng Fang, Li Shen, Tianjin Huang, Zhangyang "Atlas" Wang, Vlado Menkovski, Xiaolong Ma, Mykola Pechenizkiy, Shiwei Liu

### Effective Targeted Attacks for Adversarial Self-Supervised Learning

**Authors:** Minseon Kim, Hyeonjeong Ha, Sooel Son, Sung Ju Hwang

### Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning

**Authors:** Akhil Bagaria, Ben Abbatematteo, Omer Gottesman, Matt Corsaro, Sreehari Rammohan, George Konidaris

### Efficient Adversarial Attacks on Online Multi-agent Reinforcement Learning

**Authors:** Guanlin Liu, Lifeng LAI

### Efficient Diffusion Policies For Offline Reinforcement Learning

**Authors:** Bingyi Kang, Xiao Ma, Chao Du, Tianyu Pang, Shuicheng Yan

### Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization

**Authors:** Jinbiao Chen, Jiahai Wang, Zizhen Zhang, Zhiguang Cao, Te Ye, Siyuan Chen

### [Spotlight] Efficient Online Clustering with Moving Costs

**Authors:** Dimitris Christou, Stratis Skoulakis, Volkan Cevher

### Efficient Potential-based Exploration in Reinforcement Learning using Inverse Dynamic Bisimulation Metric

**Authors:** Yiming Wang, Ming Yang, Renzhi Dong, Binbin Sun, Furui Liu, Leong Hou U

### Efficient Uncertainty Quantification and Reduction for Over-Parameterized Neural Networks

**Authors:** Ziyi Huang, Henry Lam, Haofeng Zhang

### Embracing the chaos: analysis and diagnosis of numerical instability in variational flows

**Authors:** Zuheng Xu, Trevor Campbell

### Embroid: Unsupervised Prediction Smoothing Can Improve Few-Shot Classification

**Authors:** Neel Guha, Mayee Chen, Kush Bhatia, Azalia Mirhoseini, Frederic Sala, Christopher Ré

### [Oral] Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity

**Authors:** Tianqin Li, Ziqi Wen, Yangfan Li, Tai Sing Lee

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2B

### Energy Discrepancies: A Score-Independent Loss for Energy-Based Models

**Authors:** Tobias Schröder, Zijing Ou, Jen Lim, Yingzhen Li, Sebastian Vollmer, Andrew Duncan

### Energy Guided Diffusion for Generating Neurally Exciting Images

**Authors:** Pawel Pierzchlewicz, Konstantin Willeke, Arne Nix, Pavithra Elumalai, Kelli Restivo, Tori Shinn, Cate Nealley, Gabrielle Rodriguez, Saumil Patel, Katrin Franke, Andreas Tolias, Fabian Sinz

### Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach

**Authors:** Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, Frank Park

### Energy-Efficient Scheduling with Predictions

**Authors:** Eric Balkanski, Noemie Perivier, Clifford Stein, Hao-Ting Wei

### Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning

**Authors:** Cristina Menghini, Andrew Delworth, Stephen Bach

### Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns

**Authors:** Xin Liu, Zheng Li, Yifan Gao, Jingfeng Yang, Tianyu Cao, Zhengyang Wang, Bing Yin, Yangqiu Song

### Estimating Propensity for Causality-based Recommendation without Exposure Data

**Authors:** Zhongzhou Liu, Yuan Fang, Min Wu

### Every Parameter Matters: Ensuring the Convergence of Federated Learning with Dynamic Heterogeneous Models Reduction

**Authors:** Hanhan Zhou, Tian Lan, Guru Prasadh Venkataramani, Wenbo Ding

### Evolving Connectivity for Recurrent Spiking Neural Networks

**Authors:** Guan Wang, Yuhao Sun, Sijie Cheng, Sen Song

### Evolving Standardization for Continual Domain Generalization over Temporal Drift

**Authors:** Mixue Xie, Shuang Li, Longhui Yuan, Chi Liu, Zehui Dai

### ExPT: Synthetic Pretraining for Few-Shot Experimental Design

**Authors:** Tung Nguyen, Sudhanshu Agrawal, Aditya Grover

### Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation

**Authors:** Berivan Isik, Wei-Ning Chen, Ayfer Ozgur, Tsachy Weissman, Albert No

### Exact Verification of ReLU Neural Control Barrier Functions

**Authors:** Hongchao Zhang, Junlin Wu, Yevgeniy Vorobeychik, Andrew Clark

### Explaining Predictive Uncertainty with Information Theoretic Shapley Values

**Authors:** David Watson, Joshua O'Hara, Niek Tax, Richard Mudd, Ido Guy

### Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture

**Authors:** Galen Pogoncheff, Jacob Granley, Michael Beyeler

### Exploiting Connections between Lipschitz Structures for Certifiably Robust Deep Equilibrium Models

**Authors:** Aaron Havens, Alexandre Araujo, Siddharth Garg, Farshad Khorrami, Bin Hu

### Exploiting hidden structures in non-convex games for convergence to Nash equilibrium

**Authors:** Iosif Sakos, Panayotis Mertikopoulos, Georgios Piliouras

### Explore to Generalize in Zero-Shot RL

**Authors:** Ev Zisselman, Itai Lavie, Daniel Soudry, Aviv Tamar

### [Spotlight] Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks

**Authors:** Yaoyu Zhu, Wei Fang, Xiaodong Xie, Tiejun Huang, Zhaofei Yu

### Exploring and Interacting with the Set of Good Sparse Generalized Additive Models

**Authors:** Chudi Zhong, Zhi Chen, Jiachang Liu, Margo Seltzer, Cynthia Rudin

### Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations

**Authors:** Yu Cao, Jingrun Chen, Yixin Luo, Xiang ZHOU

### Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models

**Authors:** George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony Caterini, Eric Taylor, Gabriel Loaiza-Ganem

### FIRAL: An Active Learning Algorithm for Multinomial Logistic Regression

**Authors:** Youguang Chen, George Biros

### FLSL: Feature-level Self-supervised Learning

**Authors:** Qing Su, Anton Netchaev, Hai Li, Shihao Ji

### Fairness Aware Counterfactuals for Subgroups

**Authors:** Loukas Kavouras, Konstantinos Tsopelas, Giorgos Giannopoulos, Dimitris Sacharidis, Eleni Psaroudaki, Nikolaos Theologitis, Dimitrios Rontogiannis, Dimitris Fotakis, Ioannis Emiris

### Fantastic Robustness Measures: The Secrets of Robust Generalization

**Authors:** Hoki Kim, Jinseong Park, Yujin Choi, Jaewook Lee

### [Spotlight] Fast Approximation of Similarity Graphs with Kernel Density Estimation

**Authors:** Peter Macgregor, He Sun

### Fast Bellman Updates for Wasserstein Distributionally Robust MDPs

**Authors:** Zhuodong Yu, Ling Dai, Shaohang Xu, Siyang Gao, Chin Pang Ho

### Fast Projected Newton-like Method for Precision Matrix Estimation under Total Positivity

**Authors:** Jian-Feng CAI, Daniel Palomar, Jiaxi Ying

### Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow Shrink Trees

**Authors:** Bryan Andrews, Joseph Ramsey, Ruben Sanchez Romero, Jazmin Camchong, Erich Kummerfeld

### [Spotlight] Faster Margin Maximization Rates for Generic Optimization Methods

**Authors:** Guanghui Wang, Zihao Hu, Vidya Muthukumar, Jacob Abernethy

### FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning

**Authors:** Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, Joost van de Weijer

### Federated Learning with Manifold Regularization and Normalized Update Reaggregation

**Authors:** Xuming An, Li Shen, Han Hu, Yong Luo

### Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration

**Authors:** Qi-wei Wang, Da-Wei Zhou, Yi-Kai Zhang, De-Chuan Zhan, Han-Jia Ye

### Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation

**Authors:** Hongcheng Wang, Andy Guan Hong Chen, Xiaoqi Li, Mingdong Wu, Hao Dong

### Finding Local Minima Efficiently in Decentralized Optimization

**Authors:** Wenhan Xian, Heng Huang

### [Spotlight] Fine-Grained Human Feedback Gives Better Rewards for Language Model Training

**Authors:** Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj (Raj) Ammanabrolu, Noah Smith, Mari Ostendorf, Hannaneh Hajishirzi

### Fine-Grained Theoretical Analysis of Federated Zeroth-Order Optimization

**Authors:** Jun Chen, Hong Chen, Bin Gu, Hao Deng

### Finite Population Regression Adjustment and Non-asymptotic Guarantees for Treatment Effect Estimation

**Authors:** Mehrdad Ghadiri, David Arbour, Tung Mai, Cameron Musco, Anup Rao

### Finite-Time Analysis of Single-Timescale Actor-Critic

**Authors:** Xuyang Chen, Lin Zhao

### Flow: Per-instance Personalized Federated Learning

**Authors:** Kunjal Panchal, Sunav Choudhary, Nisarg Parikh, Lijun Zhang, Hui Guan

### Foundation Model is Efficient Multimodal Multitask Model Selector

**Authors:** fanqing meng, Wenqi Shao, zhanglin peng, Chonghe Jiang, Kaipeng Zhang, Yu Qiao, Ping Luo

### From Cloze to Comprehension: Retrofitting Pre-trained Masked Language Models to Pre-trained Machine Reader

**Authors:** Weiwen Xu, Xin Li, Wenxuan Zhang, Meng Zhou, Wai Lam, Luo Si, Lidong Bing

### [Spotlight] From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces

**Authors:** Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, Kristina N Toutanova

### GUST: Combinatorial Generalization by Unsupervised Grouping with Neuronal Coherence

**Authors:** Hao Zheng, Hui Lin, Rong Zhao

### Gaussian Membership Inference Privacy

**Authors:** Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

### Gaussian Mixture Solvers for Diffusion Models

**Authors:** Hanzhong Guo, Cheng Lu, Fan Bao, Tianyu Pang, Shuicheng Yan, Chao Du, Chongxuan LI

### [Spotlight] Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data

**Authors:** Praveen Venkatesh, Corbett Bennett, Sam Gale, Tamina Ramirez, Greggory Heller, Severine Durand, Shawn Olsen, Stefan Mihalas

### Gaussian Process Probes (GPP) for Uncertainty-Aware Probing

**Authors:** Zi Wang, Alexander Ku, Jason Baldridge, Tom Griffiths, Been Kim

### [Spotlight] Generalization in the Face of Adaptivity: A Bayesian Perspective

**Authors:** Moshe Shenfeld, Katrina Ligett

### Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models

**Authors:** Beier Zhu, Kaihua Tang, QIANRU SUN, Hanwang Zhang

### Generalized Semi-Supervised Learning via Self-Supervised Feature Adaptation

**Authors:** Jiachen Liang, RuiBing Hou, Hong Chang, Bingpeng MA, Shiguang Shan, Xilin Chen

### Generalized equivalences between subsampling and ridge regularization

**Authors:** Pratik Patil, Jin-Hong Du

### Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport

**Authors:** Jaemoo Choi, Jaewoong Choi, Myungjoo Kang

### GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies

**Authors:** Takahiro Mimori, Michiaki Hamada

### GeoTMI: Predicting Quantum Chemical Property with Easy-to-Obtain Geometry via Positional Denoising

**Authors:** Hyeonsu Kim, Jeheon Woo, SEONGHWAN KIM, Seokhyun Moon, Jun Hyeong Kim, Woo Youn Kim

### Geometric Transformer with Interatomic Positional Encoding

**Authors:** Yusong Wang, Shaoning Li, Tong Wang, Bin Shao, Nanning Zheng, Tie-Yan Liu

### Global Optimality in Bivariate Gradient-based DAG Learning

**Authors:** Chang Deng, Kevin Bello, Pradeep Ravikumar, Bryon Aragam

### Globally injective and bijective neural operators

**Authors:** Takashi Furuya, Michael Puthawala, Matti Lassas, Maarten V. de Hoop

### Gradient-Free Kernel Stein Discrepancy

**Authors:** Matthew Fisher, Chris Oates

### Grammar Prompting for Domain-Specific Language Generation with  Large Language Models

**Authors:** Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A. Saurous, Yoon Kim

### Graph Contrastive Learning with Stable and Scalable Spectral Encoding

**Authors:** Deyu Bo, Yuan Fang, Yang Liu, Chuan Shi

### Greedy Pruning with Group Lasso Provably Generalizes for Matrix Sensing

**Authors:** Nived Rajaraman, Fnu Devvrit, Aryan Mokhtari, Kannan Ramchandran

### Group Robust Classification Without Any Group Information

**Authors:** Christos Tsirigotis, Joao Monteiro, Pau Rodriguez, David Vazquez, Aaron Courville

### HeadSculpt: Crafting 3D Head Avatars with Text

**Authors:** Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, Kwan-Yee K. Wong

### HiBug: On Human-Interpretable Model Debug

**Authors:** Muxi Chen, YU LI, Qiang Xu

### HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation

**Authors:** Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, David Bull

### [Spotlight] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality

**Authors:** Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, Jun Zhu

### Hierarchical Randomized Smoothing

**Authors:** Yan Scholten, Jan Schuchardt, Aleksandar Bojchevski, Stephan Günnemann

### Hierarchical VAEs provide a normative account of motion processing in the primate brain

**Authors:** Hadi Vafaii, Jacob Yates, Daniel Butts

### [Spotlight] Hierarchical clustering with dot products recovers hidden tree structure

**Authors:** Annie Gray, Alexander Modell, Patrick Rubin-Delanchy, Nick Whiteley

### High Precision Causal Model Evaluation with Conditional Randomization

**Authors:** Chao Ma, Cheng Zhang

### HotBEV: Hardware-oriented Transformer-based Multi-View 3D Detector for BEV Perception

**Authors:** Peiyan Dong, Zhenglun Kong, Xin Meng, Pinrui Yu, Yifan Gong, Geng Yuan, Hao Tang, Yanzhi Wang

### How a Student becomes a Teacher: learning and forgetting through Spectral methods

**Authors:** Lorenzo Giambagli, Lorenzo Buffoni, Lorenzo Chicchi, Duccio Fanelli

### How many samples are needed to leverage smoothness?

**Authors:** Vivien Cabannes, Stefano Vigogna

### HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork

**Authors:** Bipasha Sen, Gaurav Singh, Aditya Agarwal, Rohith Agaram, Madhava Krishna, Srinath Sridhar

### [Spotlight] HyTrel: Hypergraph-enhanced  Tabular Data Representation Learning

**Authors:** Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang, George Karypis

### Hypervolume Maximization: A Geometric View of Pareto Set Learning

**Authors:** Xiaoyuan Zhang, Xi Lin, Bo Xue, Yifan Chen, Qingfu Zhang

### Hypothesis Selection with Memory Constraints

**Authors:** Maryam Aliakbarpour, Mark Bun, Adam Smith

### IDEA: An Invariant Perspective for Efficient Domain Adaptive Image Retrieval

**Authors:** Haixin Wang, Hao Wu, Jinan Sun, Shikun Zhang, Chong Chen, Xian-Sheng Hua, Xiao Luo

### Idempotent Learned Image Compression with Right-Inverse

**Authors:** Yanghao Li, Tongda Xu, Yan Wang, Jingjing Liu, Ya-Qin Zhang

### Identifiable Contrastive Learning with Automatic Feature Importance Discovery

**Authors:** Qi Zhang, Yifei Wang, Yisen Wang

### Im-Promptu: In-Context Composition from Image Prompts

**Authors:** Bhishma Dedhia, Michael Chang, Jake Snell, Tom Griffiths, Niraj Jha

### Imitation Learning from Vague Feedback

**Authors:** Xin-Qiang Cai, Yu-Jie Zhang, Chao-Kai Chiang, Masashi Sugiyama

### Implicit Convolutional Kernels for Steerable CNNs

**Authors:** Maksim Zhdanov, Nico Hoffmann, Gabriele Cesa

### Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics

**Authors:** Mathias Schreiner, Ole Winther, Simon Olsson

### Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning

**Authors:** Hanlin Zhu, Paria Rashidinejad, Jiantao Jiao

### Improved Bayesian Regret Bounds for Thompson Sampling in Reinforcement Learning

**Authors:** Ahmadreza Moradipari, Mohammad Pedramfar, Modjtaba Shokrian Zini, Vaneet Aggarwal

### [Spotlight] Improved Frequency Estimation Algorithms with and without Predictions

**Authors:** Anders Aamand, Justin Chen, Huy Nguyen, Sandeep Silwal, Ali Vakilian

### Improving Graph Matching with Positional Reconstruction Encoder-Decoder Network

**Authors:** Yixiao Zhou, Ruiqi Jia, Hongxiang Lin, Hefeng Quan, Yumeng Zhao, Xiaoqing Lyu

### Improving the Knowledge Gradient Algorithm

**Authors:** Le Yang, Siyang Gao, Chin Pang Ho

### [Spotlight] In-Context Impersonation Reveals Large Language Models' Strengths and Biases

**Authors:** Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, Zeynep Akata

### [Spotlight] In-Context Learning Unlocked for Diffusion Models

**Authors:** Zhendong Wang, Yifan Jiang, Yadong Lu, yelong shen, Pengcheng He, Weizhu Chen, Zhangyang "Atlas" Wang, Mingyuan Zhou

### Incentives in Federated Learning: Equilibria, Dynamics, and Mechanisms for Welfare Maximization

**Authors:** Aniket Murhekar, Zhuowen Yuan, Bhaskar Ray Chaudhury, Bo Li, Ruta Mehta

### Incentivized Communication for Federated Bandits

**Authors:** Zhepei Wei, Chuanhao Li, Haifeng Xu, Hongning Wang

### Incomplete Multimodality-Diffused Emotion Recognition

**Authors:** Yuanzhi Wang, Yong Li, Zhen Cui

### Inconsistency, Instability, and Generalization Gap of Deep Neural Network Training

**Authors:** Rie Johnson, Tong Zhang

### [Spotlight] Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI

**Authors:** Aditya Chattopadhyay, Ryan Pilgrim, Rene Vidal

### Information Maximizing Curriculum: A Curriculum-Based Approach for Learning Versatile Skills

**Authors:** Denis Blessing, Onur Celik, Xiaogang Jia, Moritz Reuss, Maximilian Li, Rudolf Lioutikov, Gerhard Neumann

### Information Theoretic Lower Bounds for Information Theoretic Upper Bounds

**Authors:** Roi Livni

### Inner Product-based Neural Network Similarity

**Authors:** Wei Chen, Zichen Miao, Qiang Qiu

### Instructing Goal-Conditioned Reinforcement Learning Agents with Temporal Logic Objectives

**Authors:** Wenjie Qiu, Wensen Mao, He Zhu

### Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts

**Authors:** Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, Irina Piontkovskaya

### Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation

**Authors:** Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, LINGMING ZHANG

### Iteratively Learn Diverse Strategies with State Distance Information

**Authors:** Wei Fu, Weihua Du, Jingwei Li, Sunli Chen, Jingzhao Zhang, YI WU

### Joint processing of linguistic properties in brains and language models

**Authors:** SUBBAREDDY OOTA, Manish Gupta, Mariya Toneva

### Keep Various Trajectories: Promoting Exploration of Ensemble Policies in Continuous Control

**Authors:** Chao Li, Chen GONG, Qiang He, Xinwen Hou

### Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors

**Authors:** Yong Liu, Chenyu Li, Jianmin Wang, Mingsheng Long

### LANCE: Stress-testing Visual Models by Generating Language-guided Counterfactual Images

**Authors:** Viraj Prabhu, Sriram Yenamandra, Prithvijit Chattopadhyay, Judy Hoffman

### LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings

**Authors:** Ningyi Liao, Siqiang Luo, Xiang Li, Jieming Shi

### Label-efficient Segmentation via Affinity Propagation

**Authors:** Wentong Li, Yuqian Yuan, Song Wang, Wenyu Liu, Dongqi Tang, Jian liu, Jianke Zhu, Lei Zhang

### Langevin Quasi-Monte Carlo

**Authors:** Sifan Liu

### Language Model Tokenizers Introduce Unfairness Between Languages

**Authors:** Aleksandar Petrov, Emanuele La Malfa, Philip Torr, Adel Bibi

### Language Models are Weak Learners

**Authors:** Hariharan Manikandan, Yiding Jiang, J. Zico Kolter

### Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding

**Authors:** George Ma, Yifei Wang, Yisen Wang

### Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning

**Authors:** Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang

### Large Language Models are Visual Reasoning Coordinators

**Authors:** Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, Ziwei Liu

### Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering

**Authors:** Noah Hollmann, Samuel Müller, Frank Hutter

### Large language models transition from integrating across position-yoked, exponential windows to structure-yoked, power-law windows

**Authors:** David Skrill, Samuel Norman-Haignere

### Learning Energy-based Model via Dual-MCMC Teaching

**Authors:** Jiali Cui, Tian Han

### [Spotlight] Learning Functional Transduction

**Authors:** Mathieu Chalvidal, Thomas Serre, Rufin VanRullen

### Learning Interpretable Low-dimensional Representation via Physical Symmetry

**Authors:** Xuanjie Liu, Daniel Chin, Yichen Huang, Gus Xia

### Learning Invariant Representations with a Nonparametric Nadaraya-Watson Head

**Authors:** Alan Wang, Minh Nguyen, Mert Sabuncu

### Learning Large Graph Property Prediction via Graph Segment Training

**Authors:** Kaidi Cao, Mangpo Phothilimtha, Sami Abu-El-Haija, Dustin Zelle, Yanqi Zhou, Charith Mendis, Jure Leskovec, Bryan Perozzi

### Learning Large-Scale MTP$_2$ Gaussian Graphical Models via Bridge-Block Decomposition

**Authors:** Xiwen WANG, Jiaxi Ying, Daniel Palomar

### [Oral] Learning Linear Causal Representations from Interventions under General Nonlinear Mixing

**Authors:** Simon Buchholz, Goutham Rajendran, Elan Rosenfeld, Bryon Aragam, Bernhard Schölkopf, Pradeep Ravikumar

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2C

### Learning Neural Implicit through Volume Rendering with Attentive Depth Fusion Priors

**Authors:** Pengchong Hu, Zhizhong Han

### Learning Sample Difficulty from Pre-trained Models for Reliable Prediction

**Authors:** Peng Cui, Dan Zhang, Zhijie Deng, Yinpeng Dong, Jun Zhu

### Learning Space-Time Continuous Latent Neural PDEs from Partially Observed States

**Authors:** Valerii Iakovlev, Markus Heinonen, Harri Lähdesmäki

### Learning Unseen Modality Interaction

**Authors:** Yunhua Zhang, Hazel Doughty, Cees Snoek

### Learning and Collusion in Multi-unit Auctions

**Authors:** Simina Branzei, Mahsa Derakhshan, Negin Golrezaei, Yanjun Han

### Learning the Efficient Frontier

**Authors:** Philippe Chatigny, Ivan Sergienko, Ryan Ferguson, Jordan Weir, Maxime Bergeron

### Learning threshold neurons via edge of stability

**Authors:** Kwangjun Ahn, Sebastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, Yi Zhang

### Learning to Discover Skills through Guidance

**Authors:** HYUNSEUNG KIM, BYUNG KUN LEE, Hojoon Lee, Dongyoon Hwang, Sejik Park, Kyushik Min, Jaegul Choo

### [Spotlight] Learning to Receive Help: Intervention-Aware Concept Embedding Models

**Authors:** Mateo Espinosa Zarlenga, Katie Collins, Krishnamurthy Dvijotham, Adrian Weller, Zohreh Shams, Mateja Jamnik

### Learning via Wasserstein-Based High Probability Generalisation Bounds

**Authors:** Paul Viallard, Maxime Haddouche, Umut Simsekli, Benjamin Guedj

### Lightweight Vision Transformer with Bidirectional Interaction

**Authors:** Qihang Fan, Huaibo Huang, Xiaoqiang Zhou, Ran He

### Likelihood Ratio Confidence Sets for Sequential Decision Making

**Authors:** Nicolas Emmenegger, Mojmir Mutny, Andreas Krause

### [Oral] Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

**Authors:** Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, Gal Chechik

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2B

### [Spotlight] LinkerNet: Fragment Poses and Linker Co-Design with 3D Equivariant Diffusion

**Authors:** Jiaqi Guan, Xingang Peng, PeiQi Jiang, Yunan Luo, Jian Peng, Jianzhu Ma

### Live Graph Lab: Towards Open, Dynamic and Real Transaction Graphs with NFT

**Authors:** Zhen Zhang, Bingqiao Luo, Shengliang Lu, Bingsheng He

### Long-Term Fairness with Unknown Dynamics

**Authors:** Tongxin Yin, Reilly Raab, Mingyan Liu, Yang Liu

### Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping

**Authors:** Feng Zhang, Ming Tian, Zhiqiang Li, Bin Xu, Qingbo Lu, Changxin Gao, Nong Sang

### Loss Decoupling for Task-Agnostic Continual Learning

**Authors:** Yan-Shuo Liang, Wu-Jun Li

### LuminAIRe: Illumination-Aware Conditional Image Repainting for Lighting-Realistic Generation

**Authors:** Jiajun Tang, Haofeng Zhong, Shuchen Weng, Boxin Shi

### M$^{2}$SODAI: Multi-Modal Maritime Object Detection Dataset With RGB and Hyperspectral Image Sensors

**Authors:** Jonggyu Jang, Sangwoo Oh, Youjin Kim, Dongmin Seo, Youngchol Choi, Hyun Jong Yang

### MG-ViT: A Multi-Granularity Method for Compact and Efficient Vision Transformers

**Authors:** Yu Zhang, Yepeng Liu, Duoqian Miao, Qi Zhang, Yiwei Shi, Liang Hu

### MMGP: a Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under nonparametrized geometrical variability

**Authors:** Fabien Casenave, Brian Staber, Xavier Roynard

### Many-body Approximation for Non-negative Tensors

**Authors:** Kazu Ghalamkari, Mahito Sugiyama, Yoshinobu Kawahara

### Mask Propagation for Efficient Video Semantic Segmentation

**Authors:** Yuetian Weng, Mingfei Han, Haoyu He, Mingjie Li, Lina Yao, Xiaojun Chang, Bohan Zhuang

### Masked Image Residual Learning for Scaling Deeper Vision Transformers

**Authors:** Guoxi Huang, Hongtao Fu, Adrian G. Bors

### [Spotlight] Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration

**Authors:** Zhihan Liu, Miao Lu, WEI XIONG, Han Zhong, Hao Hu, Shenao Zhang, Sirui Zheng, Zhuoran Yang, Zhaoran Wang

### MeGraph: Capturing Long-Range Interactions by Alternating Local and Hierarchical Aggregation on Multi-Scaled Graph Hierarchy

**Authors:** Honghua Dong, Jiawei Xu, Yu Yang, Rui Zhao, Shiwen Wu, Chun Yuan, Xiu Li, Chris Maddison, Lei Han

### [Spotlight] Mean-field Langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction

**Authors:** Taiji Suzuki, Denny Wu, Atsushi Nitanda

### Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias

**Authors:** Zhongwei Wan, Che Liu, Mi Zhang, Jie Fu, Benyou Wang, Sibo Cheng, Lei Ma, César Quilodrán-Casas, Rossella Arcucci

### Meet in the Middle: A New Pre-training Paradigm

**Authors:** Anh Nguyen, Nikos Karampatziakis, Weizhu Chen

### Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization

**Authors:** Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, Dongsoo Lee

### Mip-Grid: Anti-aliased Grid Representations for Neural Radiance Fields

**Authors:** Seungtae Nam, Daniel Rho, Jong Hwan Ko, Eunbyung Park

### Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals

**Authors:** Tam Nguyen, Tan Nguyen, Richard Baraniuk

### Mitigating the Effect of Incidental Correlations on Part-based Learning

**Authors:** Gaurav Bhatt, Deepayan Das, Leonid Sigal, Vineeth N Balasubramanian

### MixFormerV2: Efficient Fully Transformer Tracking

**Authors:** Yutao Cui, Tianhui Song, Gangshan Wu, Limin Wang

### Model-Free Reinforcement Learning with the Decision-Estimation Coefficient

**Authors:** Dylan J Foster, Noah Golowich, Jian Qian, Alexander Rakhlin, Ayush Sekhari

### Modeling Dynamics over Meshes with Gauge Equivariant Nonlinear Message Passing

**Authors:** Jung Yeon Park, Lawson Wong, Robin Walters

### Module-wise Training of Neural Networks via the Minimizing Movement Scheme

**Authors:** Skander Karkar, Ibrahim Ayed, Emmanuel de Bézenac, Patrick Gallinari

### Momentum Provably Improves Error Feedback!

**Authors:** Ilyas Fatkhullin, Alexander Tyurin, Peter Richtarik

### [Oral] Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture

**Authors:** Dan Fu, Simran Arora, Jessica Grogan, Isys Johnson, Evan Sabri Eyuboglu, Armin Thomas, Benjamin Spector, Michael Poli, Atri Rudra, Christopher Ré

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2A

### MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining

**Authors:** Jacob Portes, Alexander Trott, Sam Havens, DANIEL KING, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, Jonathan Frankle

### MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data

**Authors:** Tianyu Liu, Yuge Wang, Rex Ying, Hongyu Zhao

### Multi-Agent Learning with Heterogeneous Linear Contextual Bandits

**Authors:** Anh Do, Thanh Nguyen-Tang, Raman Arora

### Multi-Modal Inverse Constrained Reinforcement Learning from a Mixture of Demonstrations

**Authors:** Guanren Qiao, Guiliang Liu, Pascal Poupart, Zhiqiang Xu

### [Spotlight] Multi-Object Representation Learning via Feature Connectivity and Object-Centric Regularization

**Authors:** Alex Foo, Wynne Hsu, Mong Li Lee

### Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation

**Authors:** Jia-Xing Zhong, Ta-Ying Cheng, Yuhang He, Kai Lu, Kaichen Zhou, Andrew Markham, Niki Trigoni

### MultiFusion: Fusing Pre-Trained Models for Multi-Lingual, Multi-Modal Image Generation

**Authors:** Marco Bellagente, Manuel Brack, Hannah Teufel, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, Andres Felipe Cruz-Salinas, Patrick Schramowski, Kristian Kersting, Samuel Weinbach

### Mutual Information Regularized Offline Reinforcement Learning

**Authors:** Xiao Ma, Bingyi Kang, Zhongwen Xu, Min Lin, Shuicheng Yan

### NAS-X: Neural Adaptive Smoothing via Twisting

**Authors:** Dieterich Lawson, Michael Li, Scott Linderman

### Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Approach for Object Detection

**Authors:** Taehyeon Kim, Eric Lin, Junu Lee, Christian Lau, Vaikkunth Mugunthan

### NeRF-IBVS: Visual Servo Based on NeRF for Visual Localization and Navigation

**Authors:** Yuanze Wang, Yichao Yan, Dianxi Shi, Wenhan Zhu, Jianqiang Xia, Tan Jeff, Songchang Jin, KE GAO, XIAOBO LI, Xiaokang Yang

### Near-Optimal $k$-Clustering in the Sliding Window Model

**Authors:** David Woodruff, Peilin Zhong, Samson Zhou

### Nearest Neighbour with Bandit Feedback

**Authors:** Stephen Pasteris, Chris Hicks, Vasilios Mavroudis

### [Oral] Nearly Tight Bounds For Differentially Private Multiway Cut

**Authors:** Mina Dalirrooyfard, Slobodan Mitrovic, Yuriy Nevmyvaka

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2D

### Neural Circuits for Fast Poisson Compressed Sensing in the Olfactory Bulb

**Authors:** Jacob Zavatone-Veth, Paul Masset, William Tong, Joseph D. Zak, Venkatesh Murthy, Cengiz Pehlevan

### Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity

**Authors:** Joel Ye, Jennifer Collinger, Leila Wehbe, Robert Gaunt

### Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions

**Authors:** Ruofan Wu, Jiawei Qiao, Mingzhe Wu, Wen Yu, Ming Zheng, Tengfei LIU, Tianyi Zhang, Weiqiang Wang

### Neural Lyapunov Control for Discrete-Time Systems

**Authors:** Junlin Wu, Andrew Clark, Yiannis Kantaros, Yevgeniy Vorobeychik

### Neural Priming for Sample-Efficient Adaptation

**Authors:** Matthew Wallingford, Vivek Ramanujan, Alex Fang, Aditya Kusupati, Roozbeh Mottaghi, Aniruddha Kembhavi, Ludwig Schmidt, Ali Farhadi

### Neural Sculpting: Uncovering hierarchically modular task structure in neural networks through pruning and network analysis

**Authors:** Shreyas Malakarjun Patil, Loizos Michael, Constantine Dovrolis

### Neural approximation of Wasserstein distance via a universal architecture for symmetric and factorwise group invariant functions

**Authors:** Samantha Chen, Yusu Wang

### Neuro-symbolic Learning Yielding Logical Constraints

**Authors:** Zenan Li, Yunpeng Huang, Zhaoyu Li, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, Jian Lu

### New Bounds for Hyperparameter Tuning of Regression Problems Across Instances

**Authors:** Maria-Florina Balcan, Anh Nguyen, Dravyansh Sharma

### [Spotlight] No Change, No Gain: Empowering Graph Neural Networks with Expected Model Change Maximization for Active Learning

**Authors:** Zixing Song, Yifei Zhang, Irwin King

### No Representation Rules Them All in Category Discovery

**Authors:** Sagar Vaze, Andrea Vedaldi, Andrew Zisserman

### No-Regret Learning in Dynamic Competition with Reference Effects Under Logit Demand

**Authors:** Mengzi Amy Guo, Donghao Ying, Javad Lavaei, Zuo-Jun Shen

### No-Regret Online Prediction with Strategic Experts

**Authors:** Omid Sadeghi, Maryam Fazel

### No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions

**Authors:** Tiancheng Jin, Junyan Liu, Chloé Rouyer, William Chang, Chen-Yu Wei, Haipeng Luo

### Noether Embedding: Efficient Learning of Temporal Regularities

**Authors:** Chi Gao, Zidong Zhou, Luping Shi

### Noise-Adaptive Thompson Sampling for Linear Contextual Bandits

**Authors:** Ruitu Xu, Yifei Min, Tianhao Wang

### [Spotlight] Non-Asymptotic Analysis of a UCB-based Top Two Algorithm

**Authors:** Marc Jourdan, Rémy Degenne

### [Spotlight] OKRidge: Scalable Optimal k-Sparse Ridge Regression

**Authors:** Jiachang Liu, Sam Rosen, Chudi Zhong, Cynthia Rudin

### Object-centric Learning with Cyclic Walks between Parts and Whole

**Authors:** Ziyu Wang, Mike Zheng Shou, Mengmi Zhang

### On Computing Pairwise Statistics with Local Differential Privacy

**Authors:** Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Adam Sealfon

### On Convergence of Polynomial Approximations to the Gaussian Mixture Entropy

**Authors:** Caleb Dahlke, Jason Pacheco

### On Proper Learnability between Average- and Worst-case Robustness

**Authors:** Vinod Raman, UNIQUE SUBEDI, Ambuj Tewari

### On student-teacher deviations in distillation: does it pay to disobey?

**Authors:** Vaishnavh Nagarajan, Aditya Menon, Srinadh Bhojanapalli, Hossein Mobahi, Sanjiv Kumar

### [Spotlight] On the Connection between Pre-training Data Diversity and Fine-tuning Robustness

**Authors:** Vivek Ramanujan, Thao Nguyen, Sewoong Oh, Ali Farhadi, Ludwig Schmidt

### On the Convergence of Encoder-only Shallow Transformers

**Authors:** Yongtao Wu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher

### On the Convergence of No-Regret Learning Dynamics in Time-Varying Games

**Authors:** Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, Tuomas Sandholm

### On the Exploitability of Instruction Tuning

**Authors:** Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

### On the Importance of Feature Separability in Predicting Out-Of-Distribution Error

**Authors:** RENCHUNZI XIE, Hongxin Wei, Lei Feng, Yuzhou Cao, Bo An

### On the Last-iterate Convergence in Time-varying Zero-sum Games: Extra Gradient Succeeds where Optimism Fails

**Authors:** Yi Feng, Hu Fu, Qun Hu, Ping Li, Ioannis Panageas, bo peng, Xiao Wang

### On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

**Authors:** Sangha Park, Jisoo Mok, Dahuin Jung, Saehyung Lee, Sungroh Yoon

### On the Relationship Between Relevance and Conflict in Online Social Link Recommendations

**Authors:** Yanbang Wang, Jon Kleinberg

### [Spotlight] On the Variance, Admissibility, and Stability of Empirical Risk Minimization

**Authors:** Gil Kur, Eli Putterman, Alexander Rakhlin

### On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective

**Authors:** Mathieu Serrurier, Franck Mamalet, Thomas FEL, Louis Béthune, Thibaut Boissin

### One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation

**Authors:** Zhiwei Hao, Jianyuan Guo, Kai Han, Yehui Tang, Han Hu, Yunhe Wang, Chang Xu

### [Spotlight] One-step differentiation of iterative algorithms

**Authors:** Jerome Bolte, Edouard Pauwels, Samuel Vaiter

### Online Ad Procurement in Non-stationary Autobidding Worlds

**Authors:** Jason Cheuk Nam Liang, Haihao Lu, Baoyu Zhou

### Online Clustering of Bandits with Misspecified User Models

**Authors:** Zhiyong Wang, Jize Xie, Xutong Liu, Shuai Li, John C.S. Lui

### Online POMDP Planning with Anytime Deterministic Guarantees

**Authors:** Moran Barenboim, Vadim Indelman

### Online Pricing for Multi-User Multi-Item Markets

**Authors:** Yigit Efe Erginbas, Thomas Courtade, Kannan Ramchandran, Soham Phade

### Optimal Excess Risk Bounds for Empirical Risk Minimization on $p$-Norm Linear Regression

**Authors:** Ayoub El Hanchi, Murat Erdogdu

### Optimal Time Complexities of Parallel Stochastic Optimization Methods Under a Fixed Computation Model

**Authors:** Alexander Tyurin, Peter Richtarik

### Optimal Transport Model Distributional Robustness

**Authors:** Van-Anh Nguyen, Trung Le, Anh Bui, Thanh-Toan Do, Dinh Phung

### [Spotlight] Optimal Transport-Guided Conditional Score-Based Diffusion Model

**Authors:** Xiang Gu, Liwei Yang, Jian Sun, Zongben Xu

### Optimal cross-learning for contextual bandits with unknown context distributions

**Authors:** Jon Schneider, Julian Zimmert

### Optimization and Bayes: A Trade-off for Overparameterized Neural Networks

**Authors:** Zhengmian Hu, Heng Huang

### Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources

**Authors:** Haotian Zheng, Qizhou Wang, Zhen Fang, Xiaobo Xia, Feng Liu, Tongliang Liu, Bo Han

### [Spotlight] Outlier-Robust Gromov-Wasserstein for Graph Data

**Authors:** Lemin Kong, Jiajin Li, Jianheng Tang, Anthony Man-Cho So

### [Spotlight] PAC Learning Linear Thresholds from Label Proportions

**Authors:** Anand Brahmbhatt, Rishi Saket, Aravindan Raghuveer

### [Spotlight] PAPR: Proximity Attention Point Rendering

**Authors:** Yanshu Zhang, Shichong Peng, Alireza Moazeni, Ke Li

### PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis

**Authors:** Ali TehraniJamsaz, Quazi Ishtiaque Mahmud, Le Chen, Nesreen K. Ahmed, Ali Jannesari

### PETAL: Physics Emulation Through Averaged Linearizations for Solving Inverse Problems

**Authors:** Jihui Jin, Etienne Ollivier, Richard Touret, Matthew McKinley, Karim Sabra, Justin Romberg

### POMDP Planning for Object Search in Partially Unknown Environment

**Authors:** Yongbo Chen, Hanna Kurniawati

### PTQD: Accurate Post-Training Quantization for Diffusion Models

**Authors:** Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, Bohan Zhuang

### PUe: Biased Positive-Unlabeled Learning Enhancement by Causal Inference

**Authors:** Xutao Wang, Hanting Chen, Tianyu Guo, Yunhe Wang

### Pairwise Causality Guided Transformers for Event Sequences

**Authors:** Xiao Shou, Debarun Bhattacharjya, Tian Gao, Dharmashankar Subramanian, Oktie Hassanzadeh, Kristin P Bennett

### Parameterizing Context: Unleashing the Power of Parameter-Efficient Fine-Tuning and In-Context Tuning for Continual Table Semantic Parsing

**Authors:** Yongrui Chen, Shenyu Zhang, Guilin Qi, Xinnan Guo

### Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage

**Authors:** Yuheng Jia, Fuchao Yang, Yongqiang Dong

### Passive learning of active causal strategies in agents and language models

**Authors:** Andrew Lampinen, Stephanie Chan, Ishita Dasgupta, Andrew Nam, Jane Wang

### Phase diagram of early training dynamics in deep neural networks: effect of the learning rate, depth, and width

**Authors:** Dayal Singh Kalra, Maissam Barkeshli

### PoET: A generative model of protein families as sequences-of-sequences

**Authors:** Timothy Truong Jr, Tristan Bepler

### Pointwise uncertainty quantification for sparse variational Gaussian process regression with a Brownian motion prior

**Authors:** Luke Travis, Kolyan Ray

### Polyhedron Attention Module: Learning Adaptive-order Interactions

**Authors:** Tan Zhu, Fei Dou, Xinyu Wang, Jin Lu, Jinbo Bi

### Post Hoc Explanations of Language Models Can Improve Language Models

**Authors:** Satyapriya Krishna, Jiaqi Ma, Dylan Slack, Asma Ghandeharioun, Sameer Singh, Himabindu Lakkaraju

### Posthoc privacy guarantees for collaborative inference with modified Propose-Test-Release

**Authors:** Abhishek Singh, Praneeth Vepakomma, Vivek Sharma, Ramesh Raskar

### Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning

**Authors:** Jialong Wu, Haoyu Ma, Chaoyi Deng, Mingsheng Long

### Preference-grounded Token-level Guidance for Language Model Fine-tuning

**Authors:** Shentao Yang, Shujian Zhang, Congying Xia, Yihao Feng, Caiming Xiong, Mingyuan Zhou

### Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression

**Authors:** Allan Raventós, Mansheej Paul, Feng Chen, Surya Ganguli

### [Oral] Privacy Auditing with One (1) Training Run

**Authors:** Thomas Steinke, Milad Nasr, Matthew Jagielski

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2D

### [Oral] Private Everlasting Prediction

**Authors:** Moni Naor, Kobbi Nissim, Uri Stemmer, Chao Yan

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2D

### Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantisation.

**Authors:** Chris Subia-Waud, Srinandan Dasmahapatra

### Projection Regret: Reducing Background Bias for Novelty Detection via Diffusion Models

**Authors:** Sungik Choi, Hankook Lee, Honglak Lee, Moontae Lee

### ProteinNPT: Improving protein property prediction and design with non-parametric transformers

**Authors:** Pascal Notin, Ruben Weitzman, Debora Marks, Yarin Gal

### Provable Advantage of Curriculum Learning on Parity Targets with Mixed Inputs

**Authors:** Emmanuel Abbe, Elisabetta Cornacchia, Aryo Lotfi

### Pruning vs Quantization: Which is Better?

**Authors:** Andrey Kuzmin, Markus Nagel, Mart van Baalen, Arash Behboodi, Tijmen Blankevoort

### [Oral] QLoRA: Efficient Finetuning of Quantized LLMs

**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2A

### [Spotlight] QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution

**Authors:** Haotong Qin, Yulun Zhang, Yifu Ding, Yifan liu, Xianglong Liu, Martin Danelljan, Fisher Yu

### Quantifying the Cost of Learning in Queueing Systems

**Authors:** Daniel Freund, Thodoris Lykouris, Wentao Weng

### Random-Access Infinite Context Length for Transformers

**Authors:** Amirkeivan Mohtashami, Martin Jaggi

### Re-Think and Re-Design Graph Neural Networks in Spaces of Continuous Graph Diffusion Functionals

**Authors:** Tingting Dan, Jiaqi Ding, Ziquan Wei, Shahar Kovalsky, Minjeong Kim, Won Hwa Kim, Guorong Wu

### ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence

**Authors:** Ben Dai, Yixuan Qiu

### ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction

**Authors:** Yixun Liang, Hao He, Yingcong Chen

### Real-World Image Super-Resolution as Multi-Task Learning

**Authors:** Wenlong Zhang, Xiaohui Li, Guangyuan SHI, Xiangyu Chen, Yu Qiao, Xiaoyun Zhang, Xiao-Ming Wu, Chao Dong

### Real3D-AD: A Dataset of Point Cloud Anomaly Detection

**Authors:** Jiaqi Liu, Guoyang Xie, Ruitao Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, Feng Zheng

### Recommender Systems with Generative Retrieval

**Authors:** Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, Maciej Kula, Ed Chi, Mahesh Sathiamoorthy

### Recovering Simultaneously Structured Data via Non-Convex Iteratively Reweighted Least Squares

**Authors:** Christian Kümmerle, Johannes Maly

### Recurrent Temporal Revision Graph Networks

**Authors:** Yizhou Chen, Anxiang Zeng, Qingtao Yu, Kerui Zhang, Cao Yuanpeng, Kangle Wu, Guangda Huzhang, Han Yu, Zhiming Zhou

### Reference-Based POMDPs

**Authors:** Edward Kim, Yohan Karunanayake, Hanna Kurniawati

### Refined Mechanism Design for Approximately Structured Priors via Active Regression

**Authors:** Christos Boutsikas, Petros Drineas, Marios Mertzanidis, Alexandros Psomas, Paritosh Verma

### Regularity as Intrinsic Reward for Free Play

**Authors:** Cansu Sancaktar, Justus Piater, Georg Martius

### Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models

**Authors:** Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, Kimin Lee

### Reliable learning in challenging environments

**Authors:** Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

### Removing Hidden Confounding in Recommendation: A Unified Multi-Task Learning Approach

**Authors:** Haoxuan Li, Kunhan Wu, Chunyuan Zheng, Yanghao Xiao, Hao Wang, Zhi Geng, Fuli Feng, Xiangnan He, Peng Wu

### Replicability in Reinforcement Learning

**Authors:** Amin Karbasi, Grigoris Velegkas, Lin Yang, Felix Zhou

### Replicable Clustering

**Authors:** Hossein Esfandiari, Amin Karbasi, Vahab Mirrokni, Grigoris Velegkas, Felix Zhou

### Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning

**Authors:** Francesca Bartolucci, Emmanuel de Bézenac, Bogdan Raonic, Roberto Molinaro, Siddhartha Mishra, Rima Alaifari

### Representational Strengths and Limitations of Transformers

**Authors:** Clayton Sanford, Daniel Hsu, Matus Telgarsky

### Resetting the Optimizer in Deep RL: An Empirical Study

**Authors:** Kavosh Asadi, Rasool Fakoor, Shoham Sabach

### Retaining Beneficial Information from Detrimental Data for Neural Network Repair

**Authors:** Long-Kai Huang, Peilin Zhao, Junzhou Huang, Sinno Pan

### Rethinking Gauss-Newton for learning over-parameterized models

**Authors:** Michael Arbel, Romain Menegaux, Pierre Wolinski

### Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition

**Authors:** Divin Yan, Gengchen Wei, Chen Yang, Shengzhong Zhang, zengfeng Huang

### Reusable Slotwise Mechanisms

**Authors:** Trang Nguyen, Amin Mansouri, Kanika Madan, Khuong Duy Nguyen, Kartik Ahuja, Dianbo Liu, Yoshua Bengio

### Reversible and irreversible bracket-based dynamics for deep graph neural networks

**Authors:** Anthony Gruber, Kookjin Lee, Nathaniel Trask

### Revisit the Power of Vanilla Knowledge Distillation: from Small Scale to Large Scale

**Authors:** Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang

### Revisiting Implicit Differentiation for Learning Problems in Optimal Control

**Authors:** Ming Xu, Timothy L. Molloy, Stephen Gould

### Revisiting Out-of-distribution Robustness in NLP: Benchmarks, Analysis, and LLMs Evaluations

**Authors:** Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, FangYuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, Maosong Sun

### Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective

**Authors:** Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, Han Zhao

### Riemannian Projection-free Online Learning

**Authors:** Zihao Hu, Guanghui Wang, Jacob Abernethy

### Robust Knowledge Transfer in Tiered Reinforcement Learning

**Authors:** Jiawei Huang, Niao He

### Robust Learning with Progressive Data Expansion Against Spurious Correlation

**Authors:** Yihe Deng, Yu Yang, Baharan Mirzasoleiman, Quanquan Gu

### Robust Lipschitz Bandits to Adversarial Corruptions

**Authors:** Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee

### Robust Second-Order Nonconvex Optimization and Its Application to Low Rank Matrix Sensing

**Authors:** Shuyao Li, Yu Cheng, Ilias Diakonikolas, Jelena Diakonikolas, Rong Ge, Stephen Wright

### Robustness Guarantees for Adversarially Trained Neural Networks

**Authors:** Poorya Mianjy, Raman Arora

### [Oral] Rotating Features for Object Discovery

**Authors:** Sindy Löwe, Phillip Lippe, Francesco Locatello, Max Welling

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2B

### SALSA VERDE: a machine learning attack on LWE with sparse small secrets

**Authors:** Cathy Li, Emily Wenger, Zeyuan Allen-Zhu, Francois Charton, Kristin E. Lauter

### SAMoSSA:  Multivariate Singular Spectrum Analysis with Stochastic Autoregressive Noise

**Authors:** Abdullah Alomar, Munther Dahleh, Sean Mann, Devavrat Shah

### [Spotlight] SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs

**Authors:** Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin Murphy, Alexander Hauptmann, Lu Jiang

### SaVeNet: A Scalable Vector Network for Enhanced Molecular Representation Learning

**Authors:** Sarp Aykent, Tian Xia

### Sampling from Structured Log-Concave Distributions via a Soft-Threshold Dikin Walk

**Authors:** Oren Mangoubi, Nisheeth K. Vishnoi

### Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation

**Authors:** Jinpeng Chen, Runmin Cong, Yuxuan LUO, Horace Ip, Sam Kwong

### Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels

**Authors:** Zhen Liu, ma peitian, Dongliang Chen, Wenbin Pei, Qianli Ma

### ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection

**Authors:** Zhongzhan Huang, Pan Zhou, Shuicheng Yan, Liang Lin

### [Oral] Scaling Data-Constrained Language Models

**Authors:** Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin Raffel

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2A

### Scaling laws for language encoding models in fMRI

**Authors:** Richard Antonello, Aditya Vaidya, Alexander Huth

### Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion

**Authors:** Ethan Pronovost, Meghana Reddy Ganesina, Noureldin Hendy, Zeyu Wang, Andres Morales, Kai Wang, Nick Roy

### [Spotlight] Schema-learning and rebinding as mechanisms of in-context learning and emergence

**Authors:** Sivaramakrishnan Swaminathan, Antoine Dedieu, Rajkumar Vasudeva Raju, Murray Shanahan, Miguel Lazaro-Gredilla, Dileep George

### [Spotlight] Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces

**Authors:** Sungbin Lim, EUN BI YOON, Taehyun Byun, Taewon Kang, Seungwoo Kim, Kyungjae Lee, Sungjoon Choi

### Score-based Source Separation with Applications to Digital Communication Signals

**Authors:** Tejas Jayashankar, Gary C.F. Lee, Alejandro Lancho, Amir Weiss, Yury Polyanskiy, Gregory Wornell

### Segment Anything in 3D with NeRFs

**Authors:** Jiazhong Cen, Zanwei Zhou, Jiemin Fang, chen yang, Wei Shen, Lingxi Xie, Dongsheng Jiang, XIAOPENG ZHANG, Qi Tian

### Selectivity Drives Productivity: Efficient Dataset Pruning for Enhanced Transfer Learning

**Authors:** Yihua Zhang, Yimeng Zhang, Aochuan Chen, jinghan jia, Jiancheng Liu, Gaowen Liu, Mingyi Hong, Shiyu Chang, Sijia Liu

### Self-Adaptive Motion Tracking against On-body Displacement of Flexible Sensors

**Authors:** Chengxu Zuo, Fang Jiawei, Shihui Guo, Yipeng Qin

### Self-Consistent Velocity Matching of Probability Flows

**Authors:** Lingxiao Li, Samuel Hurault, Justin Solomon

### Self-Evaluation Guided Beam Search for Reasoning

**Authors:** Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, Michael Xie

### Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells

**Authors:** Rylan Schaeffer, Mikail Khona, Tzuhsuan Ma, Cristobal Eyzaguirre, Sanmi Koyejo, Ila Fiete

### Self-supervised Object-Centric Learning for Videos

**Authors:** Görkay Aydemir, Weidi Xie, Fatma Guney

### Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination

**Authors:** Yuchen BAI, Jean-Baptiste Durand, Grégoire Vincent, Florence Forbes

### Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots

**Authors:** Ruixiang Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, Xia Hu

### Sharpness-Aware Minimization Leads to Low-Rank Features

**Authors:** Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, Nicolas Flammarion

### [Spotlight] Should I Stop or Should I Go: Early Stopping with Heterogeneous Populations

**Authors:** Hammaad Adam, Fan Yin, Huibin Hu, Neil Tenenholtz, Lorin Crawford, Lester Mackey, Allison Koenecke

### Single-Stage Visual Query Localization in Egocentric Videos

**Authors:** Hanwen Jiang, Santhosh Kumar Ramakrishnan, Kristen Grauman

### Solving Inverse Physics Problems with Score Matching

**Authors:** Benjamin Holzschuh, Simona Vegetti, Nils Thuerey

### Sparse Deep Learning for Time Series Data: Theory and Applications

**Authors:** Mingxuan Zhang, Yan Sun, Faming Liang

### Spike-driven Transformer

**Authors:** Man Yao, JiaKui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, Guoqi Li

### Spuriosity Didn’t Kill the Classifier: Using Invariant Predictions to Harness Spurious Features

**Authors:** Cian Eastwood, Shashank Singh, Andrei L Nicolicioiu, Marin Vlastelica Pogančić, Julius von Kügelgen, Bernhard Schölkopf

### [Spotlight] Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective

**Authors:** Zeyuan Yin, Eric Xing, Zhiqiang Shen

### Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures

**Authors:** David Loiseaux, Luis Scoccola, Mathieu Carrière, Magnus Bakke Botnan, Steve OUDOT

### Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark

**Authors:** Zhengfei Kuang, Yunzhi Zhang, Hong-Xing Yu, Samir Agarwala, Elliott / Shangzhe Wu, Jiajun Wu

### Statistical Analysis of Quantum State Learning Process in Quantum Neural Networks

**Authors:** Hao-Kai Zhang, Chenghong Zhu, Mingrui Jing, Xin Wang

### Statistical and Computational Trade-off in Multi-Agent Multi-Armed Bandits

**Authors:** Filippo Vannella, Alexandre Proutiere, Jaeseong Jeong

### Strategic Classification under Unknown Personalized Manipulation

**Authors:** Han Shao, Avrim Blum, Omar Montasser

### Strategic Data Sharing between Competitors

**Authors:** Nikita Tsoy, Nikola Konstantinov

### StreamNet: Memory-Efficient Streaming Tiny Deep Learning Inference on the Microcontroller

**Authors:** Hong-Sheng Zheng, Yu-Yuan Liu, Chen-Fong Hsu, Tsung Tai Yeh

### Strong and Precise Modulation of Human Percepts via Robustified ANNs

**Authors:** Guy Gaziv, Michael Lee, James J DiCarlo

### Structure from Duplicates: Neural Inverse Graphics from a Pile of Objects

**Authors:** Tianhang Cheng, Wei-Chiu Ma, Kaiyu Guan, Antonio Torralba, Shenlong Wang

### Structured Federated Learning through Clustered Additive Modeling

**Authors:** Jie Ma, Tianyi Zhou, Guodong Long, Jing Jiang, Chengqi Zhang

### Structured State Space Models for In-Context Reinforcement Learning

**Authors:** Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Singh, Feryal Behbahani

### Subclass-Dominant Label Noise: A Counterexample for the Success of Early Stopping

**Authors:** Yingbin Bai, Zhongyi Han, Erkun Yang, Jun Yu, Bo Han, Dadong Wang, Tongliang Liu

### Survival Permanental Processes for Survival Analysis with Time-Varying Covariates

**Authors:** Hideaki Kim

### Symmetry-Informed Geometric Representation for Molecules, Proteins, and Crystalline Materials

**Authors:** Shengchao Liu, weitao Du, Yanjing Li, Zhuoxinran Li, Zhiling Zheng, Chenru Duan, Zhi-Ming Ma, Omar Yaghi, Animashree Anandkumar, Christian Borgs, Jennifer Chayes, Hongyu Guo, Jian Tang

### Synthetic Experience Replay

**Authors:** Cong Lu, Philip Ball, Yee Whye Teh, Jack Parker-Holder

### Systematic Visual Reasoning through Object-Centric Relational Abstraction

**Authors:** Taylor Webb, Shanka Subhra Mondal, Jonathan D Cohen

### TMT-VIS: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation

**Authors:** Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, Yu Qiao, Hengshuang Zhao

### Tame a Wild Camera: In-the-Wild Monocular Camera Calibration

**Authors:** Shengjie Zhu, Abhinav Kumar, Masa Hu, Xiaoming Liu

### Tanh Works Better with Asymmetry

**Authors:** Dongjin Kim, Woojeong Kim, Suhyun Kim

### Tempo Adaptation in Non-stationary Reinforcement Learning

**Authors:** Hyunin Lee, Yuhao Ding, Jongmin Lee, Ming Jin, Javad Lavaei, Somayeh Sojoudi

### TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials

**Authors:** Guillem Simeon, Gianni De Fabritiis

### Test-time Training for Matching-based Video Object Segmentation

**Authors:** Juliette Bertrand, Giorgos Kordopatis Zilos, Yannis Kalantidis, Giorgos Tolias

### TextDiffuser: Diffusion Models as Text Painters

**Authors:** Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, Furu Wei

### The Adversarial Consistency of Surrogate Risks for Binary Classification

**Authors:** Natalie Frank, Jonathan Niles-Weed

### [Spotlight] The Behavior and Convergence of Local Bayesian Optimization

**Authors:** Kaiwen Wu, Kyurae Kim, Roman Garnett, Jacob Gardner

### The Contextual Lasso: Sparse Linear Models via Deep Neural Networks

**Authors:** Ryan Thompson, Amir Dezfouli, robert kohn

### [Spotlight] The Exact Sample Complexity Gain from Invariances for Kernel Regression

**Authors:** Behrooz Tahmasebi, Stefanie Jegelka

### [Spotlight] The Goldilocks of Pragmatic Understanding: Fine-Tuning Strategy Matters for Implicature Resolution by LLMs

**Authors:** Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, Edward Grefenstette

### The Grand Illusion: The Myth of Software Portability and Implications for ML Progress.

**Authors:** Fraser Mince, Dzung Dinh, Jonas Kgomo, Neil Thompson, Sara Hooker

### [Spotlight] The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance

**Authors:** Dario Paccagnan, Marco Campi, Simone Garatti

### The Quantization Model of Neural Scaling

**Authors:** Eric Michaud, Ziming Liu, Uzay Girit, Max Tegmark

### Three Iterations of (d − 1)-WL Test Distinguish Non Isometric Clouds of d-dimensional Points

**Authors:** Valentino Delle Rose, Alexander Kozachinskiy, Cristobal Rojas, Mircea Petrache, Pablo Barceló

### Time-uniform confidence bands for the CDF under nonstationarity

**Authors:** Paul Mineiro, Steven Howard

### To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis

**Authors:** Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, Yang You

### TopP&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models

**Authors:** Pum Jun Kim, Yoojin Jang, Jisu Kim, Jaejun Yoo

### Topological Obstructions and How to Avoid Them

**Authors:** Babak Esmaeili, Robin Walters, Heiko Zimmermann, Jan-Willem van de Meent

### Towards A Richer 2D Understanding of Hands at Scale

**Authors:** Tianyi Cheng, Dandan Shan, Ayda Hassen, Richard Higgins, David Fouhey

### Towards Accelerated Model Training via Bayesian Data Selection

**Authors:** Zhijie Deng, Peng Cui, Jun Zhu

### Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity

**Authors:** Metod Jazbec, James Allingham, Dan Zhang, Eric Nalisnick

### [Spotlight] Towards Automated Circuit Discovery for Mechanistic Interpretability

**Authors:** Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso

### Towards Data-Algorithm Dependent Generalization: a Case Study on Overparameterized Linear Regression

**Authors:** Jing Xu, Jiaye Teng, Yang Yuan, Andrew Yao

### Towards Efficient Image Compression Without Autoregressive Models

**Authors:** Muhammad Salman Ali, Yeongwoong Kim, Maryam Qamar, Sung-Chang Lim, Donghyun Kim, Chaoning Zhang, Sung-Ho Bae, Hui Yong Kim

### Towards Label-free Scene Understanding by Vision Foundation Models

**Authors:** Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge ZHU, Yuexin Ma, Tongliang Liu, Wenping Wang

### Towards Stable Backdoor Purification through Feature Shift Tuning

**Authors:** Rui Min, Zeyu Qin, Li Shen, Minhao Cheng

### Towards Understanding the Dynamics of Gaussian-Stein Variational Gradient Descent

**Authors:** Tianle Liu, Promit Ghosal, Krishnakumar Balasubramanian, Natesh Pillai

### TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning

**Authors:** Shuo Sun, Molei Qin, Wentao Zhang, Haochong Xia, Chuqiao Zong, Jie Ying, Yonggang Xie, Lingxuan Zhao, Xinrun Wang, Bo An

### Train 'n Trade: Foundations of Parameter Markets

**Authors:** Tzu-Heng Huang, Harit Vishwakarma, Frederic Sala

### Training Energy-Based Normalizing Flow with Score-Matching Objectives

**Authors:** Chen-Hao Chao, Wei-Fang Sun, Yen-Chang Hsu, Zsolt Kira, Chun-Yi Lee

### Training neural operators to preserve invariant measures of chaotic attractors

**Authors:** Ruoxi Jiang, Peter Y. Lu, Elena Orlova, Rebecca Willett

### Training on Foveated Images Improves Robustness to Adversarial Attacks

**Authors:** Muhammad Shah, Aqsa Kashaf, Bhiksha Raj

### Transformer-based Planning for Symbolic Regression

**Authors:** Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, Chandan Reddy

### Tuning Multi-mode Token-level Prompt Alignment across Modalities

**Authors:** Dongsheng Wang, Miaoge Li, Xinyang Liu, MingSheng Xu, Bo Chen, Hanwang Zhang

### Two Heads are Better Than One: A Simple Exploration Framework for Efficient Multi-Agent Reinforcement Learning

**Authors:** Jiahui Li, Kun Kuang, Baoxiang Wang, Xingchen Li, Fei Wu, Jun Xiao, Long Chen

### Two-Stage Predict+Optimize for MILPs with Unknown Parameters in Constraints

**Authors:** Xinyi Hu, Jasper Lee, Jimmy Lee

### UltraRE: Enhancing RecEraser for Recommendation Unlearning via Error Decomposition

**Authors:** Yuyuan Li, Chaochao Chen, Yizhao Zhang, Weiming Liu, Lingjuan Lyu, Xiaolin Zheng, Dan Meng, Jun Wang

### Unbiased Compression Saves Communication in Distributed Optimization: When and How Much?

**Authors:** Yutong He, Xinmeng Huang, Kun Yuan

### Unbiased learning of deep generative models with structured discrete representations

**Authors:** Henry C Bendekgey, Gabe Hope, Erik Sudderth

### Uncertainty-Aware Instance Reweighting for Off-Policy Learning

**Authors:** Xiaoying Zhang, Junpu Chen, Hongning Wang, Hong Xie, Yang Liu, John C.S. Lui, Hang Li

### Uncoupled and Convergent Learning in Two-Player Zero-Sum Markov Games with Bandit Feedback

**Authors:** Yang Cai, Haipeng Luo, Chen-Yu Wei, Weiqiang Zheng

### Uncovering and Quantifying Social Biases in Code Generation

**Authors:** Yan Liu, Xiaokang Chen, Yan Gao, Zhe Su, Fengji Zhang, Daoguang Zan, Jian-Guang Lou, Pin-Yu Chen, Tsung-Yi Ho

### Understanding Contrastive Learning via Distributionally Robust Optimization

**Authors:** Junkang Wu, Jiawei Chen, Jiancan Wu, Wentao Shi, Xiang Wang, Xiangnan He

### Understanding, Predicting and Better Resolving Q-Value Divergence in Offline-RL

**Authors:** Yang Yue, Rui Lu, Bingyi Kang, Shiji Song, Gao Huang

### Undirected Probabilistic Model for Tensor Decomposition

**Authors:** Zerui Tao, Toshihisa Tanaka, Qibin Zhao

### [Spotlight] Unexpected Improvements to Expected Improvement for Bayesian Optimization

**Authors:** Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, Eytan Bakshy

### Uni3DETR: Unified 3D Detection Transformer

**Authors:** Zhenyu Wang, Ya-Li Li, Xi Chen, Hengshuang Zhao, Shengjin Wang

### Uniform-in-Time Wasserstein Stability Bounds for (Noisy) Stochastic Gradient Descent

**Authors:** Lingjiong Zhu, Mert Gurbuzbalaban, Anant Raj, Umut Simsekli

### [Spotlight] Unifying Predictions of Deterministic and Stochastic Physics in Mesh-reduced Space with Sequential Flow Generative Model

**Authors:** Luning Sun, Xu Han, Han Gao, Jian-Xun Wang, Liping Liu

### Universal Gradient Descent Ascent Method for Nonconvex-Nonconcave Minimax Optimization

**Authors:** Taoli Zheng, Linglingzhi Zhu, Anthony Man-Cho So, Jose Blanchet, Jiajin Li

### [Spotlight] Universal Online Learning with Gradient Variations: A Multi-layer Online Ensemble Approach

**Authors:** Yu-Hu Yan, Peng Zhao, Zhi-Hua Zhou

### Universal Prompt Tuning for Graph Neural Networks

**Authors:** Taoran Fang, Yunchao Zhang, YANG YANG, Chunping Wang, Lei Chen

### Unsupervised Behavior Extraction via Random Intent Priors

**Authors:** Hao Hu, Yiqin Yang, Jianing Ye, Ziqing Mai, Chongjie Zhang

### Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spike Camera

**Authors:** Lujie Xia, Ziluo Ding, Rui Zhao, Jiyuan Zhang, Lei Ma, Zhaofei Yu, Tiejun Huang, Ruiqin Xiong

### Unsupervised Protein-Ligand Binding Energy Prediction via Neural Euler's Rotation Equation

**Authors:** Wengong Jin, Siranush Sarkizova, Xun Chen, Nir HaCohen, Caroline Uhler

### [Oral] User-Level Differential Privacy With Few Examples Per User

**Authors:** Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, Chiyuan Zhang

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2D

### V-InFoR: A Robust Graph Neural Networks Explainer for Structurally Corrupted Graphs

**Authors:** Senzhang Wang, Jun Yin, Chaozhuo Li, Xing Xie, Jianxin Wang

### VPGTrans: Transfer Visual Prompt Generator across LLMs

**Authors:** Ao Zhang, Hao Fei, Yuan Yao, Wei Ji, Li Li, Zhiyuan Liu, Tat-Seng Chua

### Variational Inference with Gaussian Score Matching

**Authors:** Chirag Modi, Robert Gower, Charles Margossian, Yuling Yao, David Blei, Lawrence Saul

### ViSt3D: Video Stylization with 3D CNN

**Authors:** Ayush Pande, Gaurav Sharma

### Video Dynamics Prior: An Internal Learning Approach for Robust Video Enhancements

**Authors:** Gaurav Shrivastava, Ser Nam Lim, Abhinav Shrivastava

### Volume Feature Rendering for Fast Neural Radiance Field Reconstruction

**Authors:** Kang Han, Wei Xiang, Lu Yu

### Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets

**Authors:** Anirudhan Badrinath, Yannis Flet-Berliac, Allen Nie, Emma Brunskill

### What Do Deep Saliency Models Learn about Visual Attention?

**Authors:** Shi Chen, Ming Jiang, Qi Zhao

### [Spotlight] What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement.

**Authors:** ‪Yotam Alexander‬‏, Nimrod De La Vega, Noam Razin, Nadav Cohen

### What is the Inductive Bias of Flatness Regularization? A Study of Deep Matrix Factorization Models

**Authors:** Khashayar Gatmiry, Zhiyuan Li, Tengyu Ma, Sashank Reddi, Stefanie Jegelka, Ching-Yao Chuang

### [Spotlight] When Does Optimizing a Proper Loss Yield Calibration?

**Authors:** Jaroslaw Blasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran

### When Visual Prompt Tuning Meets Source-Free Domain Adaptive Semantic Segmentation

**Authors:** Xinhong Ma, Yiming Wang, Hao Liu, Tianyu Guo, Yunhe Wang

### Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?

**Authors:** Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier

### You Only Condense Once: Two Rules for Pruning Condensed Datasets

**Authors:** Yang He, Lingao Xiao, Joey Tianyi Zhou

### [Spotlight] Zero-shot causal learning

**Authors:** Hamed Nilforoshan, Michael Moor, Yusuf Roohani, Yining Chen, Anja Šurina, Michihiro Yasunaga, Sara Oblak, Jure Leskovec

### k-Median Clustering via Metric Embedding: Towards Better Initialization with Differential Privacy

**Authors:** Chenglin Fan, Ping Li, Xiaoyun Li

</details>

<details><summary><h3 style='display: inline;'> Poster Session 3: Wednesday, Dec 13, 08:45 CT</h3></summary>

### $L_2$-Uniform Stability of Randomized Learning Algorithms: Sharper Generalization Bounds and Confidence Boosting

**Authors:** Xiaotong Yuan, Ping Li

### $\texttt{TACO}$: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning

**Authors:** Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, Furong Huang

### $\varepsilon$-fractional core stability in Hedonic Games.

**Authors:** Simone Fioravanti, Michele Flammini, Bojana Kodric, Giovanna Varricchio

### 3D-IntPhys: Towards More Generalized 3D-grounded Visual Intuitive Physics under Challenging Scenes

**Authors:** Haotian Xue, Antonio Torralba, Josh Tenenbaum, Dan Yamins, Yunzhu Li, Hsiao-Yu Tung

### [Spotlight] 4D Panoptic Scene Graph Generation

**Authors:** Jingkang Yang, Jun CEN, WENXUAN PENG, Shuai Liu, Fangzhou Hong, Xiangtai Li, Kaiyang Zhou, Qifeng Chen, Ziwei Liu

### A Dual-Stream Neural Network Explains the Functional Segregation of Dorsal and Ventral Visual Pathways in Human Brains

**Authors:** Minkyu Choi, Kuan Han, Xiaokai Wang, Yizhen Zhang, Zhongming Liu

### A Heat Diffusion Perspective on Geodesic Preserving Dimensionality Reduction

**Authors:** Guillaume Huguet, Alexander Tong, Edward De Brouwer, Yanlei Zhang, Guy Wolf, Ian Adelstein, Smita Krishnaswamy

### [Spotlight] A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation

**Authors:** Thomas FEL, Victor Boutin, Louis Béthune, Remi Cadene, Mazda Moayeri, Léo Andéol, Mathieu Chalvidal, Thomas Serre

### A Long $N$-step Surrogate Stage Reward for Deep Reinforcement Learning

**Authors:** Junmin Zhong, Ruofan Wu, Jennie Si

### A Metadata-Driven Approach to Understand Graph Neural Networks

**Authors:** Ting Wei Li, Qiaozhu Mei, Jiaqi Ma

### [Spotlight] A One-Size-Fits-All Approach to Improving Randomness in Paper Assignment

**Authors:** Yixuan Xu, Steven Jecmen, Zimeng Song, Fei Fang

### [Spotlight] A Privacy-Friendly Approach to Data Valuation

**Authors:** Jiachen (Tianhao) Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, Prateek Mittal

### A Recurrent Neural Circuit Mechanism of Temporal-scaling Equivariant Representation

**Authors:** Junfeng Zuo, Xiao Liu, Ying Nian Wu, Si Wu, Wenhao Zhang

### A Simple Yet Effective Strategy to Robustify the Meta Learning Paradigm

**Authors:** Qi Wang, Yiqin Lv, yanghe feng, Zheng Xie, Jincai Huang

### A Smooth Binary Mechanism for Efficient Private Continual Observation

**Authors:** Joel Daniel Andersson, Rasmus Pagh

### [Spotlight] A Spectral Algorithm for List-Decodable Covariance Estimation in Relative Frobenius Norm

**Authors:** Ilias Diakonikolas, Daniel Kane, Jasper Lee, Ankit Pensia, Thanasis Pittas

### A Unified Approach to Count-Based Weakly Supervised Learning

**Authors:** Vinay Shukla, Zhe Zeng, Kareem Ahmed, Guy Van den Broeck

### A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm

**Authors:** Haizhou Shi, Hao Wang

### A graphon-signal analysis of graph neural networks

**Authors:** Ron Levie

### ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning

**Authors:** Mingyu Xu, Zheng Lian, Lei Feng, Bin Liu, Jianhua Tao

### ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks

**Authors:** Jongseok Park, Kyungmin Bin, Gibum Park, Sangtae Ha, Kyunghan Lee

### ATMAN: Understanding Transformer Predictions Through Memory Efficient Attention Manipulation

**Authors:** Björn Deiseroth, Mayukh Deb, Samuel Weinbach, Manuel Brack, Patrick Schramowski, Kristian Kersting

### AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models

**Authors:** Yuancheng Wang, Zeqian Ju, Xu Tan, Lei He, Zhizheng Wu, Jiang Bian, sheng zhao

### Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation

**Authors:** Xin Yuan, Pedro Savarese, Michael Maire

### Achieving $\mathcal{O}(\epsilon^{-1.5})$ Complexity in Hessian/Jacobian-free Stochastic Bilevel Optimization

**Authors:** Yifan Yang, Peiyao Xiao, Kaiyi Ji

### Active Learning-Based Species Range Estimation

**Authors:** Christian Lange, Elijah Cole, Grant Horn, Oisin Mac Aodha

### [Spotlight] Adaptive Data Analysis in a Balanced Adversarial Model

**Authors:** Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

### Adaptive Principal Component Regression with Applications to Panel Data

**Authors:** Anish Agarwal, Keegan Harris, Justin Whitehouse, Steven Wu

### Adaptive Privacy Composition for Accuracy-first Mechanisms

**Authors:** Ryan Rogers, Gennady Samorodnitsk, Steven Wu, Aaditya Ramdas

### Addressing the speed-accuracy simulation trade-off for adaptive spiking neurons

**Authors:** Luke Taylor, Andrew King, Nicol S Harper

### Adversarial Examples Might be Avoidable: The Role of Data Concentration in Adversarial Robustness

**Authors:** Ambar Pal, Jeremias Sulam, Rene Vidal

### Adversarial Self-Training Improves Robustness and Generalization for Gradual Domain Adaptation

**Authors:** Lianghe Shi, Weiwei Liu

### Advice Querying under Budget Constraint for Online Algorithms

**Authors:** Ziyad Benomar, Vianney Perchet

### AiluRus: A Scalable ViT Framework for Dense Prediction

**Authors:** Jin Li, Yaoming Wang, XIAOPENG ZHANG, Bowen Shi, Dongsheng Jiang, Chenglin Li, Wenrui Dai, Hongkai Xiong, Qi Tian

### AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation

**Authors:** Daiki E. Matsunaga, Jongmin Lee, Jaeseok Yoon, Stefanos Leonardos, Pieter Abbeel, Kee-Eung Kim

### Aligning Language Models with Human Preferences via a Bayesian Approach

**Authors:** Jiashuo WANG, Haozhao Wang, Shichao Sun, Wenjie Li

### Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation

**Authors:** Giorgio Giannone, Akash Srivastava, Ole Winther, Faez Ahmed

### [Spotlight] Alignment with human representations supports robust few-shot learning

**Authors:** Ilia Sucholutsky, Tom Griffiths

### [Spotlight] Alternation makes the adversary weaker in two-player games

**Authors:** Volkan Cevher, Ashok Cutkosky, Ali Kavis, Georgios Piliouras, Stratis Skoulakis, Luca Viano

### An Efficient Dataset Condensation Plugin and Its Application to Continual Learning

**Authors:** Enneng Yang, Li Shen, Zhenyi Wang, Tongliang Liu, Guibing Guo

### An Information Theory Perspective on Variance-Invariance-Covariance Regularization

**Authors:** Ravid Shwartz-Ziv, Randall Balestriero, Kenji Kawaguchi, Tim G. J. Rudner, Yann LeCun

### An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions

**Authors:** Mohammad Jalali, Cheuk Ting Li, Farzan Farnia

### An Iterative Self-Learning Framework for Medical Domain Generalization

**Authors:** Zhenbang Wu, Huaxiu Yao, David Liebovitz, Jimeng Sun

### [Spotlight] An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions

**Authors:** Yingtai Xiao, Guanlin He, Danfeng Zhang, Daniel Kifer

### Anytime-Competitive Reinforcement Learning with Policy Prior

**Authors:** Jianyi Yang, Pengfei Li, Tongxin Li, Adam Wierman, Shaolei Ren

### Approximate Allocation Matching for Structural Causal Bandits with Unobserved Confounders

**Authors:** Lai Wei, Muhammad Qasim Elahi, Mahsa Ghasemi, Murat Kocaoglu

### Approximation-Generalization Trade-offs under (Approximate) Group Equivariance

**Authors:** Mircea Petrache, Shubhendu Trivedi

### Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning

**Authors:** Xiaojun Guo, Yifei Wang, Zeming Wei, Yisen Wang

### Are aligned neural networks adversarially aligned?

**Authors:** Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Pang Wei Koh, Daphne Ippolito, Florian Tramer, Ludwig Schmidt

### Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment

**Authors:** Tianhe Wu, Shuwei Shi, Haoming Cai, Mingdeng Cao, Jing Xiao, Yinqiang Zheng, Yujiu Yang

### Assumption violations in causal discovery and the robustness of score matching

**Authors:** Francesco Montagna, Atalanti Mastakouri, Elias Eulig, Nicoletta Noceti, Lorenzo Rosasco, Dominik Janzing, Bryon Aragam, Francesco Locatello

### Asymptotically Optimal Quantile Pure Exploration for Infinite-Armed Bandits

**Authors:** Evelyn Xiao-Yue Gong, Mark Sellke

### [Spotlight] Attentive Transfer Entropy to Exploit Transient Emergence of Coupling Effect

**Authors:** Xiaolei Ru, XINYA ZHANG, Zijia Liu, Jack Murdoch Moore, Gang Yan

### [Spotlight] Auditing Fairness by Betting

**Authors:** Ben Chugg, Santiago Cortes-Gomez, Bryan Wilder, Aaditya Ramdas

### Augmentation-Aware Self-Supervision for Data-Efficient GAN Training

**Authors:** Liang Hou, Qi Cao, Yige Yuan, Songtao Zhao, Chongyang Ma, Siyuan Pan, Pengfei Wan, Zhongyuan Wang, Huawei Shen, Xueqi Cheng

### [Spotlight] Balancing memorization and generalization in RNNs for high performance brain-machine Interfaces

**Authors:** Joseph Costello, Hisham Temmar, Luis Cubillos, Matthew Mender, Dylan Wallace, Matt Willsey, Parag Patil, Cynthia Chestek

### Bandit Task Assignment with Unknown Processing Time

**Authors:** Shinji Ito, Daisuke Hatano, Hanna Sumita, Kei Takemura, Takuro Fukunaga, Naonori Kakimura, Ken-Ichi Kawarabayashi

### Bayes beats Cross Validation: Efficient and Accurate Ridge Regression via Expectation Maximization

**Authors:** Shu Yu Tew, Mario Boley, Daniel Schmidt

### Better Correlation and Robustness: A Distribution-Balanced Self-Supervised Learning Framework for Automatic Dialogue Evaluation

**Authors:** Peiwen Yuan, Xinglin Wang, Jiayi Shi, Bin Sun, Yiwei Li, Prof. Kan

### Beyond Exponential Graph: Communication-Efficient Topologies for Decentralized Learning via Finite-time Convergence

**Authors:** Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, Makoto Yamada

### Bi-Level Offline Policy Optimization with Limited Exploration

**Authors:** Wenzhuo Zhou

### [Spotlight] Bifurcations and loss jumps in RNN training

**Authors:** Lukas Eisenmann, Zahra Monfared, Niclas Göring, Daniel Durstewitz

### Birder: Communication-Efficient 1-bit Adaptive Optimizer for Practical Distributed DNN Training

**Authors:** Hanyang Peng, Shuang Qin, Yue Yu, Jin Wang, Hui Wang, Ge Li

### Block Coordinate Plug-and-Play Methods for Blind Inverse Problems

**Authors:** Weijie Gan, shirin shoushtari, Yuyang Hu, Jiaming Liu, Hongyu An, Ulugbek Kamilov

### Block Low-Rank Preconditioner with Shared Basis for Stochastic Optimization

**Authors:** Jui-Nan Yen, Sai Surya Duvvuri, Inderjit Dhillon, Cho-Jui Hsieh

### Blocked Collaborative Bandits: Online Collaborative Filtering with Per-Item Budget Constraints

**Authors:** Soumyabrata Pal, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

### [Spotlight] Blockwise Parallel Transformers for Large Context Models

**Authors:** Hao Liu, Pieter Abbeel

### Boundary Guided Learning-Free Semantic Control with Diffusion Models

**Authors:** Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan

### [Oral] Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models

**Authors:** Andrew Luo, Maggie Henderson, Leila Wehbe, Michael Tarr

**Oral Presentation:** We, Dec 13, 08:00 -- Oral 3A

### Brain-like Flexible Visual Inference by Harnessing Feedback Feedforward Alignment

**Authors:** Tahereh Toosi, Elias Issa

### Brant: Foundation Model for Intracranial Neural Signal

**Authors:** Daoze Zhang, Zhizhang Yuan, YANG YANG, Junru Chen, Jingjing Wang, Yafeng Li

### Bridging the Domain Gap: Self-Supervised 3D Scene Understanding with Foundation Models

**Authors:** Zhimin Chen, Longlong Jing, Yingwei Li, Bing Li

### Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders

**Authors:** Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzcinski, Adam Dziedzic

### Byzantine-Tolerant Methods for Distributed Variational Inequalities

**Authors:** Nazarii Tupitsa, Abdulla Jasem Almansoori, Yanlin Wu, Martin Takac, Karthik Nandakumar, Samuel Horváth, Eduard Gorbunov

### C-Disentanglement: Discovering Causally-Independent Generative Factors under  an Inductive Bias of Confounder

**Authors:** Xiaoyu Liu, Jiaxin Yuan, Bang An, Yuancheng Xu, Yifan Yang, Furong Huang

### CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning

**Authors:** Charles Guille-Escuret, Pau Rodriguez, David Vazquez, Ioannis Mitliagkas, Joao Monteiro

### CBD: A Certified Backdoor Detector Based on Local Dominant Probability

**Authors:** Zhen Xiang, Zidi Xiong, Bo Li

### CD-GraB: Coordinating Distributed Example Orders for Provably Accelerated Training

**Authors:** A. Feder Cooper, Wentao Guo, Duc Khiem Pham, Tiancheng Yuan, Charlie Ruan, Yucheng Lu, Christopher De Sa

### [Spotlight] CODA: Generalizing to Open and Unseen Domains with Compaction and Disambiguation

**Authors:** Chaoqi Chen, Luyao Tang, Yue Huang, Xiaoguang Han, Yizhou Yu

### CORL: Research-oriented Deep Offline Reinforcement Learning Library

**Authors:** Denis Tarasov, Alexander Nikulin, Dmitry Akimov, Vladislav Kurenkov, Sergey Kolesnikov

### CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation

**Authors:** Yexiong Lin, Yu Yao, Xiaolong Shi, Mingming Gong, Xu Shen, Dong Xu, Tongliang Liu

### CSLP-AE: A Contrastive Split-Latent Permutation Autoencoder Framework for Zero-Shot Electroencephalography Signal Conversion

**Authors:** Anders Nørskov, Alexander Neergaard Zahid, Morten Mørup

### Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning

**Authors:** Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, Sergey Levine

### Calibration by Distribution Matching: Trainable Kernel Calibration Metrics

**Authors:** Charlie Marx, Sofian Zalouk, Stefano Ermon

### Can Language Models Teach? Teacher Explanations Improve Student Performance via Personalization

**Authors:** Swarnadeep Saha, Peter Hase, Mohit Bansal

### Cascading Bandits: Optimizing Recommendation Frequency in Delayed Feedback Environments

**Authors:** Dairui Wang, Junyu Cao, Yan Zhang, Wei Qi

### Causal Component Analysis

**Authors:** Liang Wendong, Armin Kekić, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele, Bernhard Schölkopf

### Causal Interpretation of Self-Attention in Pre-Trained Transformers

**Authors:** Raanan Rohekar, Yaniv Gurwicz, Shami Nisimov

### Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization

**Authors:** Mahyar Fazlyab, Taha Entesari, Aniket Roy, Rama Chellappa

### Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models

**Authors:** Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Jianfeng Gao

### Chasing Fairness Under Distribution Shift: A Model Weight Perturbation Approach

**Authors:** Zhimeng Jiang, Xiaotian Han, Hongye Jin, Guanchu Wang, Rui Chen, Na Zou, Xia Hu

### ChatGPT-Powered Hierarchical Comparisons for Image Classification

**Authors:** Zhiyuan Ren, Yiyang Su, Xiaoming Liu

### [Oral] Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity

**Authors:** Zijiao Chen, Jiaxin Qing, Juan Helen Zhou

**Oral Presentation:** We, Dec 13, 08:15 -- Oral 3A

### Circuit as Set of Points

**Authors:** Jialv Zou, Xinggang Wang, Jiahao Guo, Wenyu Liu, Qian Zhang, Chang Huang

### CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection

**Authors:** Yang Cao, Zeng Yihan, Hang Xu, Dan Xu

### CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift

**Authors:** Siddhartha Laghuvarapu, Zhen Lin, Jimeng Sun

### Cognitive Steering in Deep Neural Networks via Long-Range Modulatory Feedback Connections

**Authors:** Talia Konkle, George Alvarez

### Combining Behaviors with the Successor Features Keyboard

**Authors:** Wilka Carvalho Carvalho, Andre Saraiva, Angelos Filos, Andrew Lampinen, Loic Matthey, Richard L Lewis, Honglak Lee, Satinder Singh, Danilo Jimenez Rezende, Daniel Zoran

### Compositional Generalization from First Principles

**Authors:** Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, Wieland Brendel

### Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference

**Authors:** Tao Lei, Junwen Bai, Siddhartha Brahma, Joshua Ainslie, Kenton Lee, Yanqi Zhou, Nan Du, Vincent Zhao, Yuexin Wu, Bo Li, Yu Zhang, Ming-Wei Chang

### Conditional Matrix Flows for Gaussian Graphical Models

**Authors:** Marcello Massimo Negri, Fabricio Arend Torres, Volker Roth

### [Spotlight] Conditional Mutual Information for Disentangled Representations in Reinforcement Learning

**Authors:** Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah Hanna, Stefano Albrecht

### Conformal Prediction for Time Series with Modern Hopfield Networks

**Authors:** Andreas Auer, Martin Gauch, Daniel Klotz, Sepp Hochreiter

### Connecting Multi-modal Contrastive Representations

**Authors:** Zehan Wang, Yang Zhao, Xize 成, Haifeng Huang, Jiageng Liu, Aoxiong Yin, Li Tang, Linjun Li, Yongqi Wang, Ziang Zhang, Zhou Zhao

### Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent

**Authors:** Giannis Daras, Yuval Dagan, Alex Dimakis, Constantinos Daskalakis

### Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning

**Authors:** Yihang Yao, ZUXIN LIU, Zhepeng Cen, Jiacheng Zhu, Wenhao Yu, Tingnan Zhang, DING ZHAO

### Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars

**Authors:** Simon Schrodi, Danny Stoll, Binxin Ru, Rhea Sukthanker, Thomas Brox, Frank Hutter

### Context-lumpable stochastic bandits

**Authors:** Chung-Wei Lee, Qinghua Liu, Yasin Abbasi Yadkori, Chi Jin, Tor Lattimore, Csaba Szepesvari

### Contextually Affinitive Neighborhood Refinery for Deep Clustering

**Authors:** Chunlin Yu, Ye Shi, Jingya Wang

### Contrastive Moments: Unsupervised Halfspace Learning in Polynomial Time

**Authors:** Xinyuan Cao, Santosh Vempala

### Controlling Text-to-Image Diffusion by Orthogonal Finetuning

**Authors:** Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, Bernhard Schölkopf

### Convergence Analysis of Sequential Federated Learning on Heterogeneous Data

**Authors:** Yipeng Li, Xinchen Lyu

### Convergence of Actor-Critic with Multi-Layer Neural Networks

**Authors:** Haoxing Tian, Alex Olshevsky, Yannis Paschalidis

### Convolutional Neural Operators for robust and accurate learning of PDEs

**Authors:** Bogdan Raonic, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, Emmanuel de Bézenac

### Core-sets for Fair and Diverse Data Summarization

**Authors:** Sepideh Mahabadi, Stojan Trajanovski

### Corruption-Robust Offline Reinforcement Learning with General Function Approximation

**Authors:** Chenlu Ye, Rui Yang, Quanquan Gu, Tong Zhang

### Counterfactually Fair Representation

**Authors:** Zhiqun Zuo, Mahdi Khalili, Xueru Zhang

### [Spotlight] Critical Initialization of Wide and Deep Neural Networks using Partial Jacobians: General Theory and Applications

**Authors:** Darshil Doshi, Tianyu He, Andrey Gromov

### Crystal Structure Prediction by Joint Equivariant Diffusion

**Authors:** Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, Yang Liu

### Curvature Filtrations for Graph Generative Model Evaluation

**Authors:** Joshua Southern, Jeremy Wayland, Michael Bronstein, Bastian Rieck

### D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion

**Authors:** Jialin Chen, Shirley Wu, Abhijit Gupta, Rex Ying

### DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation

**Authors:** Qingkai Fang, Yan Zhou, Yang Feng

### DAW: Exploring the Better Weighting Function for Semi-supervised Semantic Segmentation

**Authors:** Rui Sun, Huayu Mai, Tianzhu Zhang, Feng Wu

### [Spotlight] DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization

**Authors:** Zhiqing Sun, Yiming Yang

### DSR: Dynamical Surface Representation as Implicit Neural Networks for Protein

**Authors:** Daiwen Sun, He Huang, Yao Li, Xinqi Gong, Qiwei Ye

### Data Quality in Imitation Learning

**Authors:** Suneel Belkhale, Yuchen Cui, Dorsa Sadigh

### Data-Centric Learning from Unlabeled Graphs with Diffusion Model

**Authors:** Gang Liu, Eric Inae, Tong Zhao, Jiaxin Xu, Tengfei Luo, Meng Jiang

### DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models

**Authors:** Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, Chunhua Shen

### Debiased and Denoised Entity Recognition from Distant Supervision

**Authors:** Haobo Wang, Yiwen Dong, Ruixuan Xiao, Fei Huang, Gang Chen, Junbo Zhao

### Debiasing Conditional Stochastic Optimization

**Authors:** Lie He, Shiva Kasiviswanathan

### Debiasing Pretrained Generative Models by Uniformly Sampling Semantic Attributes

**Authors:** Walter Gerych, Kevin Hickey, Luke Buquicchio, Kavin Chandrasekaran, Abdulaziz Alajaji, Elke A. Rundensteiner, Emmanuel Agu

### [Spotlight] Decentralized Randomly Distributed Multi-agent Multi-armed Bandit with Heterogeneous Rewards

**Authors:** Mengfan Xu, Diego Klabjan

### Decision Tree for Locally Private Estimation with Public Data

**Authors:** Yuheng Ma, Han Zhang, Yuchao Cai, Hanfang Yang

### Deep Non-line-of-sight Imaging from Under-scanning Measurements

**Authors:** Yue Li, Yueyi Zhang, Juntian Ye, Feihu Xu, Zhiwei Xiong

### Deep Stochastic Processes via Functional Markov Transition Operators

**Authors:** Jin Xu, Emilien Dupont, Kaspar Märtens, Thomas Rainforth, Yee Whye Teh

### Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks

**Authors:** Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang

### Delayed Algorithms for Distributed Stochastic Weakly Convex Optimization

**Authors:** Wenzhi Gao, Qi Deng

### Demographic Parity Constrained Minimax Optimal Regression under Linear Model

**Authors:** Kazuto Fukuchi, Jun Sakuma

### DesCo: Learning Object Recognition with Rich Language Descriptions

**Authors:** Liunian Li, Zi-Yi Dou, Nanyun Peng, Kai-Wei Chang

### Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents

**Authors:** Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian (Shawn) Ma, Yitao Liang

### Detecting hidden confounding in observational data using multiple environments

**Authors:** Rickard Karlsson, Jesse Krijthe

### Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models

**Authors:** Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhihua Zhang

### DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

**Authors:** Mintong Kang, Dawn Song, Bo Li

### Differentiable Blocks World: Qualitative 3D Decomposition by Rendering Primitives

**Authors:** Tom Monnier, Jake Austin, Angjoo Kanazawa, Alexei Efros, Mathieu Aubry

### [Oral] DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models

**Authors:** Tsun-Hsuan Johnson Wang, Juntian Zheng, Pingchuan Ma, Yilun Du, Byungchul Kim, Andrew Spielberg, Josh Tenenbaum, Chuang Gan, Daniela Rus

**Oral Presentation:** We, Dec 13, 08:30 -- Oral 3C

### Diffused Redundancy in Pre-trained Representations

**Authors:** Vedant Nanda, Till Speicher, John Dickerson, Krishna Gummadi, Soheil Feizi, Adrian Weller

### [Spotlight] Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels

**Authors:** Zebin You, Yong Zhong, Fan Bao, Jiacheng Sun, Chongxuan LI, Jun Zhu

### Diffusion-Based Probabilistic Uncertainty Estimation for Active Domain Adaptation

**Authors:** Zhekai Du, Jingjing Li

### Direct Preference-based Policy Optimization without Reward Modeling

**Authors:** Gaon An, Junhyeok Lee, Xingdong Zuo, Norio Kosaka, Kyung-Min Kim, Hyun Oh Song

### Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning

**Authors:** Wei Tang, Weijia Zhang, Min-Ling Zhang

### Discrete-Smoothness in Online Algorithms with Predictions

**Authors:** Yossi Azar, Debmalya Panigrahi, Noam Touitou

### Disentanglement via Latent Quantization

**Authors:** Kyle Hsu, William Dorrell, James Whittington, Jiajun Wu, Chelsea Finn

### Disentangling Voice and Content with Self-Supervision for Speaker Recognition

**Authors:** TIANCHI LIU, Kong Aik Lee, Qiongqiong Wang, Haizhou Li

### Distributed Personalized Empirical Risk Minimization

**Authors:** Yuyang Deng, Mohammad Mahdi Kamani, Pouria Mahdavinia, Mehrdad Mahdavi

### Distribution Learnability and Robustness

**Authors:** Shai Ben-David, Alex Bie, Gautam Kamath, Tosca Lechner

### Distributional Learning of Variational AutoEncoder: Application to Synthetic Data Generation

**Authors:** Seunghwan An, Jong-June Jeon

### Does Invariant Graph Learning via Environment Augmentation Learn Invariance?

**Authors:** Yongqiang Chen, Yatao Bian, Kaiwen Zhou, Binghui Xie, Bo Han, James Cheng

### Does Visual Pretraining Help End-to-End Reasoning?

**Authors:** Chen Sun, Calvin Luo, Xingyi Zhou, Anurag Arnab, Cordelia Schmid

### Domain Adaptive Imitation Learning with Visual Observation

**Authors:** Sungho Choi, Seungyul Han, Woojun Kim, Jongseong Chae, Whiyoung Jung, Youngchul Sung

### Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand

**Authors:** Junfeng Guo, Yiming Li, Lixu Wang, Shu-Tao Xia, Heng Huang, Cong Liu, Bo Li

### Don’t just prune by magnitude! Your mask topology is a secret weapon

**Authors:** Duc Hoang, Souvik Kundu, Shiwei Liu, Zhangyang "Atlas" Wang

### [Spotlight] DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data

**Authors:** Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, Phillip Isola

### DropCompute: simple and more robust  distributed synchronous training via compute variance reduction

**Authors:** Niv Giladi, Shahar Gottlieb, moran shkolnik, Asaf Karnieli, Ron Banner, Elad Hoffer, Kfir Y. Levy, Daniel Soudry

### [Spotlight] Dynamic Tensor Decomposition via Neural Diffusion-Reaction Processes

**Authors:** Zheng Wang, Shikai Fang, Shibo Li, Shandian Zhe

### Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion

**Authors:** Yang Liu, Feng Wang, Naiyan Wang, ZHAO-XIANG ZHANG

### Ecosystem-level Analysis of Deployed Machine Learning Reveals Homogeneous Outcomes

**Authors:** Connor Toups, Rishi Bommasani, Kathleen Creel, Sarah Bana, Dan Jurafsky, Percy Liang

### Effective Robustness against Natural Distribution Shifts for Models with Different Training Data

**Authors:** Zhouxing Shi, Nicholas Carlini, Ananth Balashankar, Ludwig Schmidt, Cho-Jui Hsieh, Alex Beutel, Yao Qin

### Efficient Batched Algorithm for Contextual Linear Bandits with Large Action Space via Soft Elimination

**Authors:** Osama Hanna, Lin Yang, Christina Fragouli

### Efficient Beam Tree Recursion

**Authors:** Jishnu Ray Chowdhury, Cornelia Caragea

### Efficient Low-rank Backpropagation for Vision Transformer Adaptation

**Authors:** Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu

### Efficient Model-Free Exploration in Low-Rank MDPs

**Authors:** Zak Mhammedi, Adam Block, Dylan J Foster, Alexander Rakhlin

### Efficient Test-Time Adaptation for Super-Resolution with Second-Order Degradation and Reconstruction

**Authors:** Zeshuai Deng, Zhuokun Chen, Shuaicheng Niu, Thomas Li, Bohan Zhuang, Mingkui Tan

### Efficient Training of Energy-Based Models Using Jarzynski Equality

**Authors:** Davide Carbone, Mengjian Hua, Simon Coste, Eric Vanden-Eijnden

### EgoDistill: Egocentric Head Motion Distillation for Efficient Video Understanding

**Authors:** Shuhan Tan, Tushar Nagarajan, Kristen Grauman

### Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization

**Authors:** Runqi Lin, Chaojian Yu, Tongliang Liu

### [Spotlight] EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought

**Authors:** Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, Ping Luo

### Emergent and Predictable Memorization in Large Language Models

**Authors:** Stella Biderman, USVSN PRASHANTH, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff

### Energy Transformer

**Authors:** Benjamin Hoover, Yuchen Liang, Bao Pham, Rameswar Panda, Hendrik Strobelt, Duen Horng Chau, Mohammed Zaki, Dmitry Krotov

### Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification

**Authors:** Jintong Gao, He Zhao, Zhuo Li, Dandan Guo

### Ensemble-based Deep Reinforcement Learning for Vehicle Routing Problems under Distribution Shift

**Authors:** YUAN JIANG, Zhiguang Cao, Yaoxin Wu, Wen Song, Jie Zhang

### [Oral] Entropic Neural Optimal Transport via Diffusion Processes

**Authors:** Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry Vetrov, Evgeny Burnaev

**Oral Presentation:** We, Dec 13, 08:15 -- Oral 3C

### Entropy-dissipation Informed Neural Network for McKean-Vlasov Type PDEs

**Authors:** Zebang Shen, Zhenfu Wang

### [Spotlight] Episodic Multi-Task Learning with Heterogeneous Neural Processes

**Authors:** Jiayi Shen, Xiantong Zhen, Qi Wang, Marcel Worring

### [Spotlight] Equivariant Neural Operator Learning with Graphon Convolution

**Authors:** Chaoran Cheng, Jian Peng

### Estimating Koopman operators with sketching to provably learn large scale dynamical systems

**Authors:** Giacomo Meanti, Antoine Chatalic, Vladimir Kostic, Pietro Novelli, Massimiliano Pontil, Lorenzo Rosasco

### [Spotlight] Evaluating and Inducing Personality in Pre-trained Language Models

**Authors:** Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, Yixin Zhu

### [Spotlight] Evaluating the Moral Beliefs Encoded in LLMs

**Authors:** Nino Scherrer, Claudia Shi, Amir Feder, David Blei

### EvoFed: Leveraging Evolutionary Strategies for Communication-Efficient Federated Learning

**Authors:** Mohammad Mahdi Rahimi, Hasnain Irshad Bhatti, Younghyun Park, Humaira Kousar, Do-Yeon Kim, Jaekyun Moon

### Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing

**Authors:** Shangshang Yang, Xiaoshan Yu, Ye Tian, Xueming Yan, Haiping Ma, Xingyi Zhang

### Experimental Designs for Heteroskedastic Variance

**Authors:** Justin Weltz, Tanner Fiez, Alexander Volfovsky, Eric Laber, Blake Mason, houssam nassif, Lalit Jain

### Explainable and Efficient Randomized Voting Rules

**Authors:** Soroush Ebadian, Aris Filos-Ratsikas, Mohamad Latifian, Nisarg Shah

### Exploiting Contextual Objects and Relations for 3D Visual Grounding

**Authors:** Li Yang, chunfeng yuan, Ziqi Zhang, Zhongang Qi, Yan Xu, Wei Liu, Ying Shan, Bing Li, Weiping Yang, Peng Li, Yan Wang, Weiming Hu

### Exploring Question Decomposition for Zero-Shot VQA

**Authors:** Zaid Khan, Vijay Kumar B G, Samuel Schulter, Manmohan Chandraker, Yun Fu

### Extracting Reward Functions from Diffusion Models

**Authors:** Felipe Nuti, Tim Franzmeyer, João Henriques

### FABind: Fast and Accurate Protein-Ligand Binding

**Authors:** Qizhi Pei, Kaiyuan Gao, Lijun Wu, Jinhua Zhu, Yingce Xia, Shufang Xie, Tao Qin, Kun He, Tie-Yan Liu, Rui Yan

### FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models

**Authors:** Hao ZHANG, Tianyuan DAI, Yanbo Xu, Yu-Wing Tai, Chi-Keung Tang

### Facilitating Graph Neural Networks with Random Walk on Simplicial Complexes

**Authors:** Cai Zhou, Xiyuan Wang, Muhan Zhang

### Fair Adaptive Experiments

**Authors:** Waverly Wei, Xinwei Ma, Jingshen Wang

### Fairly Recommending with Social Attributes: A Flexible and Controllable Optimization Approach

**Authors:** Jinqiu Jin, Haoxuan Li, Fuli Feng, Sihao Ding, Peng Wu, Xiangnan He

### Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments

**Authors:** Thanh-Dat Truong, Hoang-Quan Nguyen, Bhiksha Raj, Khoa Luu

### Fairness-guided Few-shot Prompting for Large Language Models

**Authors:** Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, Bingzhe Wu

### Fast Attention Requires Bounded Entries

**Authors:** Josh Alman, Zhao Song

### Fast Optimal Locally Private Mean Estimation via Random Projections

**Authors:** Hilal Asi, Vitaly Feldman, Jelani Nelson, Huy Nguyen, Kunal Talwar

### Fast Rank-1 Lattice Targeted Sampling for Black-box Optimization

**Authors:** Yueming LYU

### Fast and Simple Spectral Clustering in Theory and Practice

**Authors:** Peter Macgregor

### Faster Query Times for Fully Dynamic $k$-Center Clustering with Outliers

**Authors:** Leyla Biabani, Annika Hennes, Morteza Monemizadeh, Melanie Schmidt

### Faster Relative Entropy Coding with Greedy Rejection Coding

**Authors:** Gergely Flamich, Stratis Markou, José Miguel Hernández-Lobato

### Faster approximate subgraph counts with privacy

**Authors:** Dung Nguyen, Mahantesh Halappanavar, Venkatesh Srinivasan, Anil Vullikanti

### Feature Learning for Interpretable, Performant Decision Trees

**Authors:** Jack Good, Torin Kovach, Kyle Miller, Artur Dubrawski

### Feature-Learning Networks Are Consistent Across Widths At Realistic Scales

**Authors:** Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, Cengiz Pehlevan

### Federated Conditional Stochastic Optimization

**Authors:** Xidong Wu, Jianhui Sun, Zhengmian Hu, Junyi Li, Aidong Zhang, Heng Huang

### Federated Multi-Objective Learning

**Authors:** Haibo Yang, Zhuqing Liu, Jia Liu, Chaosheng Dong, Michinari Momma

### Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering

**Authors:** Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, Bill Byrne

### FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing

**Authors:** Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, Ziwei Liu

### First Order Methods with Markovian Noise: from Acceleration to Variational Inequalities

**Authors:** Aleksandr Beznosikov, Sergey Samsonov, Marina Sheshukova, Alexander Gasnikov, Alexey Naumov, Eric Moulines

### First Order Stochastic Optimization with Oblivious Noise

**Authors:** Ilias Diakonikolas, Sushrut Karmalkar, Jong Ho Park, Christos Tzamos

### First- and Second-Order Bounds for Adversarial Linear Contextual Bandits

**Authors:** Julia Olkhovskaya, Jack Mayo, Tim van Erven, Gergely Neu, Chen-Yu Wei

### Flow Factorized Representation Learning

**Authors:** Yue Song, T. Anderson Keller, Nicu Sebe, Max Welling

### Flow Matching for Scalable Simulation-Based Inference

**Authors:** Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen Green, Jakob H Macke, Bernhard Schölkopf

### For SALE: State-Action Representation Learning for Deep Reinforcement Learning

**Authors:** Scott Fujimoto, Wei-Di Chang, Edward Smith, Shixiang (Shane) Gu, Doina Precup, David Meger

### Formulating Discrete Probability Flow Through Optimal Transport

**Authors:** Pengze Zhang, Hubery Yin, Chen Li, Xiaohua Xie

### Fractal Landscapes in Policy Optimization

**Authors:** Tao Wang, Sylvia Herbert, Sicun Gao

### Frequency Domain-Based Dataset Distillation

**Authors:** Donghyeok Shin, Seungjae Shin, Il-chul Moon

### From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization

**Authors:** Yang Li, Jinpei Guo, Runzhong Wang, Junchi Yan

### [Spotlight] From Tempered to Benign Overfitting in ReLU Neural Networks

**Authors:** Guy Kornowski, Gilad Yehudai, Ohad Shamir

### Functional Renyi Differential Privacy for Generative Modeling

**Authors:** Dihong Jiang, Sun Sun, Yaoliang Yu

### [Spotlight] Future-Dependent Value-Based Off-Policy Evaluation in POMDPs

**Authors:** Masatoshi Uehara, Haruka Kiyohara, Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, Wen Sun

### GRAND-SLAMIN’ Interpretable Additive Modeling with Structural Constraints

**Authors:** Shibal Ibrahim, Gabriel Afriat, Kayhan Behdin, Rahul Mazumder

### Generalised f-Mean Aggregation for Graph Neural Networks

**Authors:** Ryan Kortvelesy, Steven D Morad, Amanda Prorok

### [Spotlight] Generalizing Importance Weighting to A Universal Solver for Distribution Shift Problems

**Authors:** Tongtong Fang, Nan Lu, Gang Niu, Masashi Sugiyama

### Generating Images with Multimodal Language Models

**Authors:** Jing Yu Koh, Daniel Fried, Russ Salakhutdinov

### Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning

**Authors:** Changyu CHEN, Ramesha Karunasena, Thanh Nguyen, Arunesh Sinha, Pradeep Varakantham

### Generative Neural Fields by Mixtures of Neural Implicit Functions

**Authors:** Tackgeun You, Mijeong Kim, Jungtaek Kim, Bohyung Han

### GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization

**Authors:** Vicente Vivanco Cepeda, Gaurav Kumar Nayak, Mubarak Shah

### Geometric Algebra Transformer

**Authors:** Johann Brehmer, Pim de Haan, Sönke Behrends, Taco Cohen

### Geometric Analysis of Matrix Sensing over Graphs

**Authors:** Haixiang Zhang, Ying Chen, Javad Lavaei

### Geometric Neural Diffusion Processes

**Authors:** Emile Mathieu, Vincent Dutordoir, Michael Hutchinson, Valentin De Bortoli, Yee Whye Teh, Richard Turner

### GlucoSynth: Generating Differentially-Private Synthetic Glucose Traces

**Authors:** Josephine Lamp, Mark Derdzinski, Christopher Hannemann, Joost van der Linden, Lu Feng, Tianhao Wang, David Evans

### Goal Driven Discovery of Distributional Differences via Language Descriptions

**Authors:** Ruiqi Zhong, Peter Zhang, Steve Li, Jinwoo Ahn, Dan Klein, Jacob Steinhardt

### Goal-Conditioned Predictive Coding for Offline Reinforcement Learning

**Authors:** Zilai Zeng, Ce Zhang, Shijie Wang, Chen Sun

### Goal-conditioned Offline Planning from Curious Exploration

**Authors:** Marco Bagatella, Georg Martius

### GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients

**Authors:** Sima Behpour, Thang Long Doan, Xin Li, Wenbin He, Liang Gou, Liu Ren

### Gradient-Based Feature Learning under Structured Data

**Authors:** Alireza Mousavi-Hosseini, Denny Wu, Taiji Suzuki, Murat Erdogdu

### Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling

**Authors:** Haotao Wang, Ziyu Jiang, Yuning You, Yan Han, Gaowen Liu, Jayanth Srinivasa, Ramana Kompella, Zhangyang "Atlas" Wang

### Grassmann Manifold Flows for Stable Shape Generation

**Authors:** Ryoma Yataka, Kazuki Hirashima, Masashi Shiraishi

### Guiding The Last Layer in Federated Learning with Pre-Trained Models

**Authors:** Gwen Legate, Nicolas Bernier, Lucas Page-Caccia, Edouard Oyallon, Eugene Belilovsky

### H-nobs: Achieving Certified Fairness and Robustness in Distributed Learning on Heterogeneous Datasets

**Authors:** Guanqiang Zhou, Ping Xu, Yue Wang, Zhi Tian

### H3T: Efficient Integration of Memory Optimization and Parallelism for Large-scale Transformer Training

**Authors:** Yuzhong Wang, Xu Han, Weilin Zhao, Guoyang Zeng, Zhiyuan Liu, Maosong Sun

### HEDNet: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds

**Authors:** Gang Zhang, Chen Junnan, Guohuan Gao, Jianmin Li, Xiaolin Hu

### HQA-Attack: Toward High Quality Black-Box Hard-Label Adversarial Attack on Text

**Authors:** Han Liu, Zhi Xu, Xiaotong Zhang, Feng Zhang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang

### Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition

**Authors:** Sara Pieri, Jose Restom, Samuel Horváth, Hisham Cholakkal

### Hardness of Low Rank Approximation of Entrywise Transformed Matrix Products

**Authors:** Tamas Sarlos, Xingyou Song, David Woodruff, Richard Zhang

### Hardware Resilience Properties of Text-Guided Image Classifiers

**Authors:** Syed Talal Wasim, Kabila Haile Soboka, Abdulrahman Mahmoud, Salman Khan, David Brooks, Gu-Yeon Wei

### Harnessing the power of choices in decision tree learning

**Authors:** Guy Blanc, Jane Lange, Chirag Pabbaraju, Colin Sullivan, Li-Yang Tan, Mo Tiwari

### [Spotlight] Hierarchically Gated Recurrent Neural Network for Sequence Modeling

**Authors:** Zhen Qin, Songlin Yang, Yiran Zhong

### High dimensional, tabular deep learning with an auxiliary knowledge graph

**Authors:** Camilo Ruiz, Hongyu Ren, Kexin Huang, Jure Leskovec

### [Spotlight] Honesty Is the Best Policy: Defining and Mitigating AI Deception

**Authors:** Francis Ward, Francesca Toni, Francesco Belardinelli, Tom Everitt

### How Re-sampling Helps for Long-Tail Learning?

**Authors:** Jiang-Xin Shi, Tong Wei, Yuke Xiang, Yu-Feng Li

### [Spotlight] How to Scale Your EMA

**Authors:** Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau Cuadros, Russell Webb

### How2comm: Communication-Efficient and Collaboration-Pragmatic Multi-Agent Perception

**Authors:** Dingkang Yang, Kun Yang, Yuzheng Wang, Jing Liu, Zhi Xu, Rongbin Yin, Peng Zhai, Lihua Zhang

### Human spatiotemporal pattern learning as probabilistic program synthesis

**Authors:** Tracey Mills, Josh Tenenbaum, Samuel Cheyette

### Human-Aligned Calibration for AI-Assisted Decision Making

**Authors:** Nina Corvelo Benz, Manuel Rodriguez

### [Oral] Human-like Few-Shot Learning via Bayesian Reasoning over Natural Language

**Authors:** Kevin Ellis

**Oral Presentation:** We, Dec 13, 08:30 -- Oral 3A

### HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models

**Authors:** CHEN CHEN, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, Eng-Siong Chng

### Hybrid Policy Optimization from Imperfect Demonstrations

**Authors:** Hanlin Yang, Chao Yu, peng sun, Siji Chen

### Hyperbolic VAE via Latent Gaussian Distributions

**Authors:** Seunghyuk Cho, Juyong Lee, Dongwoo Kim

### [Spotlight] ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets

**Authors:** Damien Teney, Yong Lin, Seong Joon Oh, Ehsan Abbasnejad

### IDRNet: Intervention-Driven Relation Network for Semantic Segmentation

**Authors:** Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, Lequan Yu

### IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers

**Authors:** Zhenglin Huang, Xiaoan Bao, Na Zhang, Qingqi Zhang, Xiao Tu, Biao Wu, Xi Yang

### ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation

**Authors:** ya sheng sun, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike

### ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation

**Authors:** Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong

### Imbalanced Mixed Linear Regression

**Authors:** Pini Zilber, Boaz Nadler

### Implicit Bias of (Stochastic) Gradient Descent for Rank-1 Linear Neural Network

**Authors:** Bochen Lyu, Zhanxing Zhu

### Implicit Bias of Gradient Descent for Two-layer ReLU and Leaky ReLU Networks on Nearly-orthogonal Data

**Authors:** Yiwen Kou, Zixiang Chen, Quanquan Gu

### Importance-aware Co-teaching for Offline Model-based Optimization

**Authors:** Ye Yuan, Can Chen, Zixuan Liu, Willie Neiswanger, Xue (Steve) Liu

### Improved Bayes Risk Can Yield Reduced Social Welfare Under Competition

**Authors:** Meena Jagadeesan, Michael Jordan, Jacob Steinhardt, Nika Haghtalab

### Improvements on Uncertainty Quantification for Node Classification via Distance Based Regularization

**Authors:** Russell Hart, Linlin Yu, Yifei Lou, Feng Chen

### Improving Few-Shot Generalization by Exploring and Exploiting Auxiliary Data

**Authors:** Alon Albalak, Colin Raffel, William Yang Wang

### Improving neural network representations using human similarity judgments

**Authors:** Lukas Muttenthaler, Lorenz Linhardt, Jonas Dippel, Robert Vandermeulen, Katherine Hermann, Andrew Lampinen, Simon Kornblith

### Improving the Privacy and Practicality of Objective Perturbation for Differentially Private Linear Learners

**Authors:** Rachel Redberg, Antti Koskela, Yu-Xiang Wang

### Information Geometry of the Retinal Representation Manifold

**Authors:** Xuehao Ding, Dongsoo Lee, Joshua Melander, George Sivulka, Surya Ganguli, Stephen Baccus

### Information-guided Planning: An Online Approach for Partially Observable Problems

**Authors:** Amokh Varma, Yehia Elkhatib, Leandro Soriano Marcolino

### Initialization Matters: Privacy-Utility Analysis of Overparameterized Neural Networks

**Authors:** Jiayuan Ye, Zhenyu Zhu, Fanghui Liu, Reza Shokri, Volkan Cevher

### Interpretable Graph Networks Formulate Universal Algebra Conjectures

**Authors:** Francesco Giannini, Stefano Fioravanti, Oguzhan Keskin, Alisia Lupidi, Lucie Charlotte Magister, Pietro Lió, Pietro Barbiero

### Interpretable and Explainable Logical Policies via Neurally Guided Symbolic Abstraction

**Authors:** Quentin Delfosse, Hikaru Shindo, Devendra Dhami, Kristian Kersting

### Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP

**Authors:** Qi Qian, Yuanhong Xu, Juhua Hu

### Intriguing Properties of Quantization at Scale

**Authors:** Arash Ahmadian, Saurabh Dash, Hongyu Chen, Bharat Venkitesh, Zhen Stephen Gou, Phil Blunsom, Ahmet Üstün, Sara Hooker

### [Spotlight] Invariant Learning via Probability of Sufficient and Necessary Causes

**Authors:** Mengyue Yang, Yonggang Zhang, Zhen Fang, Yali Du, Furui Liu, Jean-Francois Ton, Jianhong Wang, Jun Wang

### Is Distance Matrix Enough for Geometric Deep Learning?

**Authors:** Zian Li, Xiyuan Wang, Yinan Huang, Muhan Zhang

### Isometric Quotient Variational Auto-Encoders for Structure-Preserving Representation Learning

**Authors:** In Huh, changwook jeong, Jae Myung Choe, YOUNGGU KIM, Daesin Kim

### Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels

**Authors:** Zifu Wang, Xuefei Ning, Matthew Blaschko

### Jigsaw: Learning to Assemble Multiple Fractured Objects

**Authors:** Jiaxin Lu, Yifan Sun, Qixing Huang

### Joint Attribute and Model Generalization Learning for Privacy-Preserving Action Recognition

**Authors:** Duo Peng, Li Xu, Qiuhong Ke, Ping Hu, Jun Liu

### Joint Data-Task Generation for Auxiliary Learning

**Authors:** Hong Chen, Xin Wang, Yuwei Zhou, Yijian Qin, Chaoyu Guan, Wenwu Zhu

### KAKURENBO: Adaptively Hiding Samples in Deep Neural Network Training

**Authors:** Truong Thao Nguyen, Balazs Gerofi, Edgar Josafat Martinez-Noriega, François Trahay, Mohamed Wahib

### [Spotlight] Kernel Quadrature with Randomly Pivoted Cholesky

**Authors:** Ethan Epperly, Elvira Moreno

### [Spotlight] Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures

**Authors:** Runa Eschenhagen, Alexander Immer, Richard Turner, Frank Schneider, Philipp Hennig

### LIMA: Less Is More for Alignment

**Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy

### LLM-Pruner: On the Structural Pruning of Large Language Models

**Authors:** Xinyin Ma, Gongfan Fang, Xinchao Wang

### LLMScore: Unveiling the Power of Large Language Models in Text-to-Image Synthesis Evaluation

**Authors:** Yujie Lu, Xianjun Yang, Xiujun Li, Xin Eric Wang, William Yang Wang

### Label Robust and Differentially Private Linear Regression: Computational and Statistical Efficiency

**Authors:** Xiyang Liu, Prateek Jain, Weihao Kong, Sewoong Oh, Arun Suggala

### Label-Only Model Inversion Attacks via Knowledge Transfer

**Authors:** Bao-Ngoc Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man (Man) Cheung

### Language Models can Solve Computer Tasks

**Authors:** Geunwoo Kim, Pierre Baldi, Stephen McAleer

### Large Language Models can Implement Policy Iteration

**Authors:** Ethan Brooks, Logan Walls, Richard L Lewis, Satinder Singh

### Latent Diffusion for Language Generation

**Authors:** Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, Kilian Weinberger

### Latent Field Discovery in Interacting Dynamical Systems with Neural Fields

**Authors:** Miltiadis (Miltos) Kofinas, Erik Bekkers, Naveen Nagaraja, Efstratios Gavves

### Latent exploration for Reinforcement Learning

**Authors:** Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Huang, Alexander Mathis

### Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions

**Authors:** Stefano Massaroli, Michael Poli, Dan Fu, Hermann Kumbong, Rom Parnichkun, David Romero, Aman Timalsina, Quinn McIntyre, Beidi Chen, Atri Rudra, Ce Zhang, Christopher Ré, Stefano Ermon, Yoshua Bengio

### Layer-Neighbor Sampling --- Defusing Neighborhood Explosion in GNNs

**Authors:** Muhammed Fatih Balin, Ümit Çatalyürek

### Learning Dense Flow Field for Highly-accurate Cross-view Camera Localization

**Authors:** Zhenbo Song, ze xianghui, Jianfeng Lu, Yujiao Shi

### Learning Descriptive Image Captioning via Semipermeable Maximum Likelihood Estimation

**Authors:** Zihao Yue, Anwen Hu, Liang Zhang, Qin Jin

### Learning Exponential Families from Truncated Samples

**Authors:** Jane Lee, Andre Wibisono, Emmanouil Zampetakis

### Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment

**Authors:** Zihui (Sherry) Xue, Kristen Grauman

### Learning Large-scale Neural Fields via Context Pruned Meta-Learning

**Authors:** Jihoon Tack, Subin Kim, Sihyun Yu, Jaeho Lee, Jinwoo Shin, Jonathan Richard Schwarz

### Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

**Authors:** Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, Humphrey Shi

### Learning Multi-agent Behaviors from Distributed and Streaming Demonstrations

**Authors:** Shicheng Liu, Minghui Zhu

### Learning Time-Invariant Representations for Individual Neurons from Population Dynamics

**Authors:** Lu Mi, Trung Le, Tianxing He, Eli Shlizerman, Uygar Sümbül

### [Oral] Learning Transformer Programs

**Authors:** Dan Friedman, Alexander Wettig, Danqi Chen

**Oral Presentation:** We, Dec 13, 08:30 -- Oral 3B

### Learning Visual Prior via Generative Pre-Training

**Authors:** Jinheng Xie, Kai Ye, Yudong Li, Yuexiang Li, Kevin Qinghong Lin, Yefeng Zheng, Linlin Shen, Mike Zheng Shou

### Learning in the Presence of Low-dimensional Structure: A Spiked Random Matrix Perspective

**Authors:** Jimmy Ba, Murat Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu

### Learning to Augment Distributions for Out-of-distribution Detection

**Authors:** Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, Bo Han

### [Spotlight] Leveraging sparse and shared feature activations for disentangled representation learning

**Authors:** Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodolà, Stefano Soatto, Bernhard Schölkopf, Francesco Locatello

### [Spotlight] Lexinvariant Language Models

**Authors:** Qian Huang, Eric Zelikman, Sarah Chen, Yuhuai Wu, Gregory Valiant, Percy Liang

### Logarithmic Bayes Regret Bounds

**Authors:** Alexia Atsidakou, Branislav Kveton, Sumeet Katariya, Constantine Caramanis, Sujay Sanghavi

### Lossy Image Compression with Conditional Diffusion Models

**Authors:** Ruihan Yang, Stephan Mandt

### Low-shot Object Learning with Mutual Exclusivity Bias

**Authors:** Anh Thai, Ahmad Humayun, Stefan Stojanov, Zixuan Huang, Bikram Boote, James Rehg

### MAG-GNN: Reinforcement Learning Boosted Graph Neural Network

**Authors:** Lecheng Kong, Jiarui Feng, Hao Liu, Dacheng Tao, Yixin Chen, Muhan Zhang

### MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers

**Authors:** LILI YU, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis

### [Spotlight] MGDD: A Meta Generator for Fast Dataset Distillation

**Authors:** Songhua Liu, Xinchao Wang

### [Spotlight] MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting

**Authors:** Felix Biggs, Antonin Schrab, Arthur Gretton

### Machine learning detects terminal singularities

**Authors:** Tom Coates, Alexander Kasprzyk, Sara Veneziale

### Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning

**Authors:** Baohao Liao, Shaomu Tan, Christof Monz

### [Spotlight] MeCo: Zero-Shot NAS with One Data and Single Forward Pass via Minimum Eigenvalue of Correlation

**Authors:** Tangyu Jiang, Haodi Wang, Rongfang Bie

### Meta-learning families of plasticity rules in recurrent spiking networks using simulation-based inference

**Authors:** Basile Confavreux, Poornima Ramesh, Pedro Goncalves, Jakob H Macke, Tim Vogels

### Minimax-Optimal Location Estimation

**Authors:** Shivam Gupta, Jasper Lee, Eric Price, Paul Valiant

### Minimum Description Length and Generalization Guarantees for Representation Learning

**Authors:** Milad Sefidgaran, Abdellatif Zaidi, Piotr Krasnowski

### Minimum norm interpolation by perceptra: Explicit regularization and implicit bias

**Authors:** Jiyoung Park, Ian Pelakh, Stephan Wojtowytsch

### Mitigating Test-Time Bias for Fair Image Retrieval

**Authors:** Fanjie Kong, Shuai Yuan, Weituo Hao, Ricardo Henao

### [Spotlight] Mitigating the Popularity Bias of  Graph Collaborative Filtering: A Dimensional Collapse Perspective

**Authors:** Yifei Zhang, Hao Zhu, yankai Chen, Zixing Song, Piotr Koniusz, Irwin King

### Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models

**Authors:** Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, WUYOU XIAO, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou

### Mnemosyne: Learning to Train Transformers with Transformers

**Authors:** Deepali Jain, Krzysztof M Choromanski, Kumar Avinava Dubey, Sumeet Singh, Vikas Sindhwani, Tingnan Zhang, Jie Tan

### Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM

**Authors:** Ziba Parsons, Fei Dou, Houyi Du, Zheng Song, Jin Lu

### Model-free Posterior Sampling via Learning Rate Randomization

**Authors:** Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey Naumov, Pierre Perrault, Michal Valko, Pierre Ménard

### Modulated Neural ODEs

**Authors:** Ilze Amanda Auzina, Çağatay Yıldız, Sara Magliacane, Matthias Bethge, Efstratios Gavves

### Multi-Player Zero-Sum Markov Games with Networked Separable Interactions

**Authors:** Chanwoo Park, Kaiqing Zhang, Asuman Ozdaglar

### Multi-Step Generalized Policy Improvement by Leveraging Approximate Models

**Authors:** Lucas N. Alegre, Ana Bazzan, Ann Nowe, Bruno da Silva

### Multiply Robust Federated Estimation of Targeted Average Treatment Effects

**Authors:** Larry Han, Zhu Shen, Jose Zubizarreta

### Multitask Learning with No Regret: from Improved Confidence Bounds to Active Learning

**Authors:** Pier Giuseppe Sessa, Pierre Laforgue, Nicolò Cesa-Bianchi, Andreas Krause

### Mutual-Information Regularized Multi-Agent Policy Iteration

**Authors:** Wang, Deheng Ye, Zongqing Lu

### NAR-Former V2: Rethinking Transformer for Universal Neural Network Representation Learning

**Authors:** Yun Yi, Haokui Zhang, Rong Xiao, Nannan Wang, Xiaoyu Wang

### NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF

**Authors:** Stefan Lionar, Xiangyu Xu, Min Lin, Gim Hee Lee

### Nash Regret Guarantees for Linear Bandits

**Authors:** Ayush Sawarni, Soumyabrata Pal, Siddharth Barman

### Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment

**Authors:** Carsten Lüth, Till Bungert, Lukas Klein, Paul Jaeger

### Near-Linear Time Algorithm for the Chamfer Distance

**Authors:** Ainesh Bakshi, Piotr Indyk, Rajesh Jayaram, Sandeep Silwal, Erik Waingarten

### Networks are Slacking Off: Understanding Generalization Problem in Image Deraining

**Authors:** Jinjin Gu, Xianzheng Ma, Xiangtao Kong, Yu Qiao, Chao Dong

### Neural Functional Transformers

**Authors:** Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J. Zico Kolter, Chelsea Finn

### Neural Graph Generation from Graph Statistics

**Authors:** Kiarash Zahirnia, Yaochen Hu, Mark Coates, Oliver Schulte

### Neural Image Compression: Generalization, Robustness, and Spectral Biases

**Authors:** Kelsey Lieberman, James Diffenderfer, Charles Godfrey, Bhavya Kailkhura

### Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization

**Authors:** Haitz Sáez de Ocáriz Borde, Alvaro Arroyo, Ismael Morales, Ingmar Posner, Xiaowen Dong

### Neural Modulation for Flash Memory: An Unsupervised Learning Framework for Improved Reliability

**Authors:** Jonathan Zedaka, Elisha Halperin, Evgeny Blaichman, Amit Berman

### Neural Multi-Objective Combinatorial Optimization with Diversity Enhancement

**Authors:** Jinbiao Chen, Zizhen Zhang, Zhiguang Cao, Yaoxin Wu, Yining Ma, Te Ye, Jiahai Wang

### Neural Oscillators are Universal

**Authors:** Samuel Lanthaler, T. Konstantin Rusch, Siddhartha Mishra

### Neural Processes with Stability

**Authors:** Huafeng Liu, Liping Jing, Jian Yu

### Neural-Logic Human-Object Interaction Detection

**Authors:** Liulei Li, Jianan Wei, Wenguan Wang, Yi Yang

### NeuralGF: Unsupervised Point Normal Estimation by Learning Neural Gradient Function

**Authors:** Qing Li, Huifang Feng, Kanle Shi, Yue Gao, Yi Fang, Yu-Shen Liu, Zhizhong Han

### Non-autoregressive Machine Translation with Probabilistic Context-free Grammar

**Authors:** Shangtong Gui, Chenze Shao, Zhengrui Ma, xishan zhang, Yunji Chen, Yang Feng

### Non-stationary Experimental Design under Linear Trends

**Authors:** David Simchi-Levi, Chonghuan Wang, Zeyu Zheng

### Normalization-Equivariant Neural Networks with Application to Image Denoising

**Authors:** Sébastien Herbreteau, Emmanuel Moebel, Charles Kervrann

### NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA

**Authors:** Hyeong Kyu Choi, Seunghun Lee, Jaewon Chu, Hyunwoo Kim

### Off-Policy Evaluation for Human Feedback

**Authors:** Qitong Gao, Ge Gao, Juncheng Dong, Vahid Tarokh, Min Chi, Miroslav Pajic

### On Dynamic Programming Decompositions of Static Risk Measures in Markov Decision Processes

**Authors:** Jia Lin Hau, Erick Delage, Mohammad Ghavamzadeh, Marek Petrik

### On Robust Streaming for Learning with Experts: Algorithms and Lower Bounds

**Authors:** David Woodruff, Fred Zhang, Samson Zhou

### On Single-Index Models beyond Gaussian Data

**Authors:** Aaron Zweig, Loucas PILLAUD-VIVIEN, Joan Bruna

### On Slicing Optimality for Mutual Information

**Authors:** Ammar Fayad, Majd Ibrahim

### On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks

**Authors:** Laura F. Nern, Harsh Raj, Maurice André Georgi, Yash Sharma

### [Spotlight] On quantum backpropagation, information reuse, and cheating measurement collapse

**Authors:** Amira Abbas, Robbie King, Hsin-Yuan Huang, William J. Huggins, Ramis Movassagh, Dar Gilboa, Jarrod McClean

### On the Adversarial Robustness of Out-of-distribution Generalization Models

**Authors:** Xin Zou, Weiwei Liu

### On the Asymptotic Learning Curves of Kernel Ridge Regression under Power-law Decay

**Authors:** Yicheng Li, haobo Zhang, Qian Lin

### On the Complexity of Differentially Private Best-Arm Identification with Fixed Confidence

**Authors:** Achraf Azize, Marc Jourdan, Aymen Al Marjani, Debabrota Basu

### On the Constrained Time-Series Generation Problem

**Authors:** Andrea Coletta, Sriram Gopalakrishnan, Daniel Borrajo, Svitlana Vyetrenko

### On the Exploration of Local Significant Differences For Two-Sample Test

**Authors:** Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao

### On the Generalization Error of Stochastic Mirror Descent for Quadratically-Bounded Losses: an Improved Analysis

**Authors:** Ta Duy Nguyen, Alina Ene, Huy Nguyen

### On the Implicit Bias of Linear Equivariant Steerable Networks

**Authors:** Ziyu Chen, Wei Zhu

### On the Overlooked Structure of Stochastic Gradients

**Authors:** Zeke Xie, Qian-Yuan Tang, Mingming Sun, Ping Li

### On the Pareto Front of Multilingual Neural Machine Translation

**Authors:** Liang Chen, Shuming Ma, Dongdong Zhang, Furu Wei, Baobao Chang

### On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm

**Authors:** Qi CHEN, Changjian Shui, Ligong Han, Mario Marchand

### On the Statistical Consistency of Risk-Sensitive Bayesian Decision-Making

**Authors:** Prateek Jaiswal, Harsha Honnappa, Vinayak Rao

### On the Trade-off of Intra-/Inter-class Diversity for Supervised Pre-training

**Authors:** Jieyu Zhang, Bohan Wang, Zhengyu Hu, Pang Wei Koh, Alexander Ratner

### On the spectral bias of two-layer linear networks

**Authors:** Aditya  Vardhan Varre, Maria-Luiza Vladarean, Loucas PILLAUD-VIVIEN, Nicolas Flammarion

### One Less Reason for Filter Pruning: Gaining Free Adversarial Robustness with Structured Grouped Kernel Pruning

**Authors:** Shaochen (Henry) Zhong, Zaichuan You, Jiamu Zhang, Sebastian Zhao, Zachary LeClaire, Zirui Liu, Daochen Zha, Vipin Chaudhary, Shuai Xu, Xia Hu

### OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling

**Authors:** yifan zhang, Qingsong Wen, xue wang, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan

### [Spotlight] Online Label Shift: Optimal Dynamic Regret meets Practical Algorithms

**Authors:** Dheeraj Baby, Saurabh Garg, Tzu-Ching Yen, Sivaraman Balakrishnan, Zachary Lipton, Yu-Xiang Wang

### Online Learning under Adversarial Nonlinear Constraints

**Authors:** Pavel Kolev, Georg Martius, Michael Muehlebach

### Online Performative Gradient Descent for Learning Nash Equilibria in Decision-Dependent Games

**Authors:** Zihan Zhu, Ethan Fang, Zhuoran Yang

### Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation

**Authors:** Chaofan Ma, Yang Yuhuan, Chen Ju, Fei Zhang, Ya Zhang, Yanfeng Wang

### OpenGSL: A Comprehensive Benchmark for Graph Structure Learning

**Authors:** Zhou Zhiyao, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu Tan, Daochen Zha, Yan Feng, Chun Chen, Can Wang

### OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding

**Authors:** Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, Hao Su

### Opening the Vocabulary of Egocentric Actions

**Authors:** Dibyadip Chatterjee, Fadime Sener, Shugao Ma, Angela Yao

### Optimal Preconditioning and Fisher Adaptive Langevin Sampling

**Authors:** Michalis Titsias

### Optimal Rates for Bandit Nonstochastic Control

**Authors:** Y. Jennifer Sun, Stephen Newman, Elad Hazan

### Optimal Transport for Treatment Effect Estimation

**Authors:** Hao Wang, Jiajun Fan, Zhichao Chen, Haoxuan Li, Weiming Liu, Tianqiao Liu, Quanyu Dai, Yichao Wang, Zhenhua Dong, Ruiming Tang

### Optimal Treatment Allocation for Efficient Policy Evaluation in Sequential Decision Making

**Authors:** Ting Li, Chengchun Shi, Jianing Wang, Fan Zhou, hongtu zhu

### Optimization of Inter-group criteria for clustering with minimum size constraints

**Authors:** Eduardo Laber, Lucas Murtinho

### Optimization or Architecture: How to Hack Kalman Filtering

**Authors:** Ido Greenberg, Netanel Yannay, Shie Mannor

### Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal

**Authors:** Leah Chrestien, Stefan Edelkamp, Antonin Komenda, Tomas Pevny

### Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation

**Authors:** Yilin Lyu, Liyuan Wang, Xingxing Zhang, Zicheng Sun, Hang Su, Jun Zhu, Liping Jing

### [Spotlight] PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

**Authors:** Phillip Lippe, Bas Veeling, Paris Perdikaris, Richard Turner, Johannes Brandstetter

### PICProp: Physics-Informed Confidence Propagation for Uncertainty Quantification

**Authors:** Qianli Shen, Wai Hoh Tang, Zhun Deng, Apostolos Psaros, Kenji Kawaguchi

### PaintSeg: Painting Pixels for Training-free Segmentation

**Authors:** Xiang Li, Chung-Ching Lin, Yinpeng Chen, Zicheng Liu, Jinglu Wang, Rita Singh, Bhiksha Raj

### Parameterizing Non-Parametric Meta-Reinforcement Learning Tasks via Subtask Decomposition

**Authors:** Suyoung Lee, Myungsik Cho, Youngchul Sung

### Partial Multi-Label Learning with Probabilistic Graphical Disambiguation

**Authors:** Jun-Yi Hang, Min-Ling Zhang

### Payoff-based Learning with Matrix Multiplicative Weights in Quantum Games

**Authors:** Kyriakos Lotidis, Panayotis Mertikopoulos, Nicholas Bambos, Jose Blanchet

### Penalising the biases in norm regularisation enforces sparsity

**Authors:** Etienne Boursier, Nicolas Flammarion

### Perceptual adjustment queries and an inverted measurement paradigm for low-rank metric learning

**Authors:** Austin Xu, Andrew McRae, Jingyan Wang, Mark Davenport, Ashwin Pananjady

### Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms

**Authors:** Yashaswini Murthy, Mehrdad Moharrami, R. Srikant

### Perturbation Towards Easy Samples Improves Targeted Adversarial Transferability

**Authors:** Junqi Gao, Biqing Qi, Yao Li, Zhichang Guo, Dong Li, Yuming Xing, Dazhi Zhang

### Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation

**Authors:** Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy

### PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change

**Authors:** Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, Subbarao Kambhampati

### Policy Gradient for Rectangular Robust Markov Decision Processes

**Authors:** Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y. Levy, Shie Mannor

### Practical Differentially Private Hyperparameter Tuning with Subsampling

**Authors:** Antti Koskela, Tejas Kulkarni

### PreDiff: Precipitation Nowcasting with Latent Diffusion Models

**Authors:** Zhihan Gao, Xingjian Shi, Boran Han, Hao Wang, Xiaoyong Jin, Danielle Maddix, Yi Zhu, Mu Li, Yuyang (Bernie) Wang

### Prediction and Control in Continual Reinforcement Learning

**Authors:** Nishanth Anand, Doina Precup

### PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation

**Authors:** Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, Ziwei Liu

### [Spotlight] Private estimation algorithms for stochastic block models and mixture models

**Authors:** Hongjie Chen, Vincent Cohen-Addad, Tommaso d’Orsi, Alessandro Epasto, Jacob Imola, David Steurer, Stefan Tiegel

### Prompt-augmented Temporal Point Process for Streaming Event Sequence

**Authors:** Siqiao Xue, Yan Wang, Zhixuan Chu, Xiaoming Shi, Caigao JIANG, Hongyan Hao, Gangwei Jiang, Xiaoyun Feng, James Zhang, Jun Zhou

### PromptIR: Prompting for All-in-One Image Restoration

**Authors:** Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan

### Propagating Knowledge Updates to LMs Through Distillation

**Authors:** Shankar Padmanabhan, Yasumasa Onoe, Michael Zhang, Greg Durrett, Eunsol Choi

### Proportional Response: Contextual Bandits for Simple and Cumulative Regret Minimization

**Authors:** Sanath Kumar Krishnamurthy, Ruohan Zhan, Susan Athey, Emma Brunskill

### [Spotlight] Protein Design with Guided Discrete Diffusion

**Authors:** Nate Gruver, Samuel Stanton, Nathan Frey, Tim G. J. Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, Andrew Wilson

### Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval

**Authors:** Hao Li, Jingkuan Song, Lianli Gao, Xiaosu Zhu, Hengtao Shen

### [Spotlight] Provably Bounding Neural Network Preimages

**Authors:** Suhas Kotha, Christopher Brix, J. Zico Kolter, Krishnamurthy Dvijotham, Huan Zhang

### [Spotlight] Provably Fast Finite Particle Variants of SVGD via Virtual Particle Stochastic Approximation

**Authors:** Aniket Das, Dheeraj Nagaraj

### Provably Robust Temporal Difference Learning for Heavy-Tailed Rewards

**Authors:** Semih Cayci, Atilla Eryilmaz

### Pseudo-Likelihood Inference

**Authors:** Theo Gruner, Boris Belousov, Fabio Muratore, Daniel Palenicek, Jan Peters

### [Spotlight] QuACK: Accelerating Gradient-Based Quantum Optimization with Koopman Operator Learning

**Authors:** Di Luo, Jiayu Shen, Rumen Dangovski, Marin Soljacic

### Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework

**Authors:** Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng, Nicholas Allen, Randy Auerbach, Faisal Mahmood, Russ Salakhutdinov, Louis-Philippe Morency

### Quantum speedups for stochastic optimization

**Authors:** Aaron Sidford, Chenyi Zhang

### R-divergence for Estimating Model-oriented Distribution Discrepancy

**Authors:** Zhilin Zhao, Longbing Cao

### RADAR: Robust AI-Text Detection via Adversarial Learning

**Authors:** Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

### REFINE: A Fine-Grained Medication Recommendation System Using Deep Learning and Personalized Drug Interaction Modeling

**Authors:** Suman Bhoi, Mong Li Lee, Wynne Hsu, Ngiap Chuan Tan

### RRHF: Rank Responses to Align Language Models with Human Feedback

**Authors:** Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang

### [Spotlight] Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks

**Authors:** Jules Berman, Benjamin Peherstorfer

### Randomized and Deterministic Maximin-share Approximations for Fractionally Subadditive Valuations

**Authors:** Hannaneh Akrami, Kurt Mehlhorn, Masoud Seddighin, Golnoosh Shahkarami

### [Spotlight] RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability

**Authors:** Chuning Zhu, Max Simchowitz, Siri Gadipudi, Abhishek Gupta

### Reading Relevant Feature from Global Representation Memory for Visual Object Tracking

**Authors:** Xinyu Zhou, Pinxue Guo, Lingyi Hong, Jinglun Li, Wei Zhang, Weifeng Ge, Wenqiang Zhang

### Recaptured Raw Screen Image and Video Demoiréing via Channel and Spatial Modulations

**Authors:** Yijia Cheng, Yijia Cheng, Xin Liu, Jingyu Yang

### Red Teaming Deep Neural Networks with Feature Synthesis Tools

**Authors:** Stephen Casper, Tong Bu, Yuxiao Li, Jiawei Li, Kevin Zhang, Kaivalya Hariharan, Dylan Hadfield-Menell

### ResMem: Learn what you can and memorize the rest

**Authors:** Zitong Yang, MICHAL LUKASIK, Vaishnavh Nagarajan, Zonglin Li, Ankit Rawat, Manzil Zaheer, Aditya Menon, Sanjiv Kumar

### Residual Q-Learning: Offline and Online Policy Customization without Value

**Authors:** Chenran Li, Chen Tang, Haruki Nishimura, Jean Mercat, Masayoshi TOMIZUKA, Wei Zhan

### ResoNet: Noise-Trained Physics-Informed MRI Off-Resonance Correction

**Authors:** Alfredo De Goyeneche Macaya, Shreya Ramachandran, Ke Wang, Ekin Karasan, Joseph Y. Cheng, Stella X. Yu, Michael Lustig

### ResoNet: a Physics-Informed DL Framework for Off-Resonance Correction in MRI Trained with Noise

**Authors:** Alfredo De Goyeneche Macaya, Shreya Ramachandran, Ke Wang, Ekin Karasan, Joseph Y. Cheng, Stella X. Yu, Michael Lustig

### Responsible AI (RAI) Games and Ensembles

**Authors:** Yash Gupta, Runtian Zhai, Arun Suggala, Pradeep Ravikumar

### [Spotlight] Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption

**Authors:** Yige Hong, Qiaomin Xie, Yudong Chen, Weina Wang

### Rethinking the Backward Propagation for Adversarial Transferability

**Authors:** Wang Xiaosen, Kangheng Tong, Kun He

### Reward Imputation with Sketching for Contextual Batched Bandits

**Authors:** Xiao Zhang, Ninglu Shao, Zihua Si, Jun Xu, Wenhan Wang, Hanjing Su, Ji-Rong Wen

### Reward Scale Robustness for Proximal Policy Optimization via DreamerV3 Tricks

**Authors:** Ryan Sullivan, Akarsh Kumar, Shengyi Huang, John Dickerson, Joseph Suarez

### Reward-agnostic Fine-tuning: Provable Statistical Benefits of Hybrid Reinforcement Learning

**Authors:** Gen Li, Wenhao Zhan, Jason Lee, Yuejie Chi, Yuxin Chen

### RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization

**Authors:** Siqi Shen, Chennan Ma, Chao Li, Weiquan Liu, Yongquan Fu, Songzhu Mei, Xinwang Liu, Cheng Wang

### RoboCLIP: One Demonstration is Enough to Learn Robot Policies

**Authors:** Sumedh Sontakke, Jesse Zhang, Séb Arnold, Karl Pertsch, Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, Laurent Itti

### Robust Concept Erasure via Kernelized Rate-Distortion Maximization

**Authors:** Somnath Basu Roy Chowdhury, Nicholas Monath, Kumar Avinava Dubey, Amr Ahmed, Snigdha Chaturvedi

### Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks

**Authors:** Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

### Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms

**Authors:** Alexander Bukharin, Yan Li, Yue Yu, Qingru Zhang, Zhehui Chen, Simiao Zuo, Chao Zhang, Songan Zhang, Tuo Zhao

### Robust covariance estimation with missing values and cell-wise contamination

**Authors:** Grégoire Pacreau, Karim Lounici

### Robust low-rank training via approximate orthonormal constraints

**Authors:** Dayana Savostianova, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco

### Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model

**Authors:** Amine Ouasfi, Adnane Boukhayma

### S-CLIP: Semi-supervised Vision-Language Learning using Few Specialist Captions

**Authors:** Sangwoo Mo, Minkyu Kim, Kyungmin Lee, Jinwoo Shin

### [Spotlight] SE(3) Equivariant Augmented Coupling Flows

**Authors:** Laurence Midgley, Vincent Stimper, Javier Antorán, Emile Mathieu, Bernhard Schölkopf, José Miguel Hernández-Lobato

### SEGA: Instructing Text-to-Image Models using Semantic Guidance

**Authors:** Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, Kristian Kersting

### SHAP-IQ: Unified Approximation of any-order Shapley Interactions

**Authors:** Fabian Fumagalli, Maximilian Muschalik, Patrick Kolpaczki, Eyke Hüllermeier, Barbara Hammer

### SHOT: Suppressing the Hessian along the Optimization Trajectory for Gradient-Based Meta-Learning

**Authors:** JunHoo Lee, Jayeon Yoo, Nojun Kwak

### SLIBO-Net: Floorplan Reconstruction via Slicing Box Representation with Local Geometry Regularization

**Authors:** Jheng-Wei Su, Kuei-Yu Tung, Chi-Han Peng, Peter Wonka, Hung-Kuo (James) Chu

### SLM: A Smoothed First-Order Lagrangian Method for Structured Constrained Nonconvex Optimization

**Authors:** Songtao Lu

### SLaM: Student-Label Mixing for  Distillation with Unlabeled Examples

**Authors:** Vasilis Kontonis, Fotis Iliopoulos, Khoa Trinh, Cenk Baykal, Gaurav Menghani, Erik Vee

### SNAP: Self-Supervised Neural Maps for Visual Positioning and Semantic Understanding

**Authors:** Paul-Edouard Sarlin, Eduard Trulls, Marc Pollefeys, Jan Hosang, Simon Lynen

### SOAR: Improved Indexing for Approximate Nearest Neighbor Search

**Authors:** Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar

### SOAR: Improved Quantization for Approximate Nearest Neighbor Search

**Authors:** Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar

### SOC: Semantic-Assisted  Object Cluster for Referring Video Object Segmentation

**Authors:** Zhuoyan Luo, Yicheng Xiao, Yong Liu, Shuyan Li, Yitong Wang, Yansong Tang, Xiu Li, Yujiu Yang

### STREAMER: Streaming Representation Learning and Event Segmentation in a Hierarchical Manner

**Authors:** Ramy Mounir, Sujal Vijayaraghavan, Sudeep Sarkar

### SUBP: Soft Uniform Block Pruning for 1$\times$N Sparse CNNs Multithreading Acceleration

**Authors:** JINGYANG XIANG, Siqi Li, Jun Chen, Guang Dai, Shipeng Bai, Yukai Ma, Yong Liu

### Sample Complexity of Goal-Conditioned Hierarchical Reinforcement Learning

**Authors:** Arnaud Robert, Ciara Pike-Burke, Aldo Faisal

### [Spotlight] Sample Efficient Reinforcement Learning in Mixed Systems through Augmented Samples and Its Applications to Queueing Networks

**Authors:** Honghao Wei, Xin Liu, Weina Wang, Lei Ying

### Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent RL with General Utilities

**Authors:** Donghao Ying, Yunkai Zhang, Yuhao Ding, Alec Koppel, Javad Lavaei

### [Spotlight] Scale Alone Does not Improve Mechanistic Interpretability in Vision Models

**Authors:** Roland S. Zimmermann, Thomas Klein, Wieland Brendel

### Scaling MLPs: A Tale of Inductive Bias

**Authors:** Gregor Bachmann, Sotiris Anagnostidis, Thomas Hofmann

### Scaling Riemannian Diffusion Models

**Authors:** Aaron Lou, Minkai Xu, Adam Farris, Stefano Ermon

### SceneScape: Text-Driven Consistent Scene Generation

**Authors:** Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel

### Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

**Authors:** Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava

### [Spotlight] Score-based Generative Models with Lévy Processes

**Authors:** EUN BI YOON, Keehun Park, Sungwoong Kim, Sungbin Lim

### SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process

**Authors:** Mengyu Wang, Henghui Ding, Jun Hao Liew, Jiajun Liu, Yao Zhao, Yunchao Wei

### Selectively Sharing Experiences Improves Multi-Agent Reinforcement Learning

**Authors:** Matthias Gerstgrasser, Tom Danino, Sarah Keren

### Semantic Image Synthesis with Unconditional Generator

**Authors:** JungWoo Chae, Hyunin Cho, Sooyeon Go, Kyungmook Choi, Youngjung Uh

### Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback

**Authors:** Minyoung Hwang, Gunmin Lee, Hogun Kee, Chan Woo Kim, Kyungjae Lee, Songhwai Oh

### Sheaf Hypergraph Networks

**Authors:** Iulia Duta, Giulia Cassarà, Fabrizio Silvestri, Pietro Lió

### Should Under-parameterized Student Networks Copy or Average Teacher Weights?

**Authors:** Berfin Simsek, Amire Bendjeddou, Wulfram Gerstner, Johanni Brea

### Simple, Scalable and Effective Clustering via One-Dimensional Projections

**Authors:** Moses Charikar, Monika Henzinger, Lunjia Hu, Maximilian Vötsch, Erik Waingarten

### Simplicity Bias in 1-Hidden Layer Neural Networks

**Authors:** Depen Morwani, Jatin Batra, Prateek Jain, Praneeth Netrapalli

### Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions

**Authors:** Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, Elad Hazan

### [Spotlight] Skill-it! A data-driven skills framework for understanding and training language models

**Authors:** Mayee Chen, Nicholas Roberts, Kush Bhatia, Jue WANG, Ce Zhang, Frederic Sala, Christopher Ré

### [Spotlight] SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models

**Authors:** Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, Animesh Garg

### Smooth Flipping Probability for Differential Private Sign Random Projection Methods

**Authors:** Ping Li, Xiaoyun Li

### Speculative Decoding with Big Little Decoder

**Authors:** Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael Mahoney, Amir Gholami, Kurt Keutzer

### Spiking PointNet: Spiking Neural Networks for Point Clouds

**Authors:** Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo

### [Spotlight] Squared Neural Families: A New Class of Tractable Density Models

**Authors:** Russell Tsuchida, Cheng Soon Ong, Dino Sejdinovic

### Stability-penalty-adaptive follow-the-regularized-leader: Sparsity, game-dependency, and best-of-both-worlds

**Authors:** Taira Tsuchiya, Shinji Ito, Junya Honda

### StableFDG: Style and Attention Based Learning for Federated Domain Generalization

**Authors:** Jungwuk Park, Dong-Jun Han, Jinho Kim, Shiqiang Wang, Christopher Brinton, Jaekyun Moon

### State Regularized Policy Optimization on Data with Dynamics Shift

**Authors:** Zhenghai Xue, Qingpeng Cai, Shuchang Liu, Dong Zheng, Peng Jiang, Kun Gai, Bo An

### [Spotlight] State Sequences Prediction via Fourier Transform for Representation Learning

**Authors:** Mingxuan Ye, Yufei Kuang, Jie Wang, Yang Rui, Wengang Zhou, Houqiang Li, Feng Wu

### Statistically Valid Variable Importance Assessment through Conditional Permutations

**Authors:** Ahmad CHAMMA, Denis Engemann, Bertrand Thirion

### Stochastic Approximation Approaches to Group Distributionally Robust Optimization

**Authors:** Lijun Zhang, Peng Zhao, Zhen-Hua Zhuang, Tianbao Yang, Zhi-Hua Zhou

### Stochastic Optimal Control for Collective Variable Free Sampling of Molecular Transition Paths

**Authors:** Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, Max Welling

### Strategic Distribution Shift of Interacting Agents via Coupled Gradient Flows

**Authors:** Lauren Conger, Franca Hoffmann, Eric Mazumdar, Lillian Ratliff

### Streaming Factor Trajectory Learning for Temporal Tensor Decomposition

**Authors:** Shikai Fang, Xin Yu, Shibo Li, Zheng Wang, Mike Kirby, Shandian Zhe

### [Spotlight] Streaming PCA for Markovian Data

**Authors:** Syamantak Kumar, Purnamrita Sarkar

### Structured Neural-PI Control with End-to-End Stability and Output Tracking Guarantees

**Authors:** Wenqi Cui, Yan Jiang, Baosen Zhang, Yuanyuan Shi

### StyleDrop: Text-to-Image Synthesis of Any Style

**Authors:** Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, Yuan Hao, Glenn Entis, Irina Blok, Daniel Castro Chin

### StyleGAN knows Normal, Depth, Albedo, and More

**Authors:** Anand Bhattad, Daniel McKee, Derek Hoiem, David Forsyth

### Sub-optimality of the Naive Mean Field approximation for proportional high-dimensional Linear Regression

**Authors:** Jiaze Qiu

### Switching Temporary Teachers for Semi-Supervised Semantic Segmentation

**Authors:** Jaemin Na, Jung-Woo Ha, Hyung Jin Chang, Joon Chung, Wonjun Hwang

### SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions

**Authors:** Yuseung Lee, Kunho Kim, Hyunjin Kim, Minhyuk Sung

### Synthetic-to-Real Pose Estimation with Geometric Reconstruction

**Authors:** Qiuxia Lin, Kerui Gu, Linlin Yang, Angela Yao

### Tanimoto Random Features for Scalable Molecular Machine Learning

**Authors:** Austin Tripp, Sergio Bacallado, Sukriti Singh, José Miguel Hernández-Lobato

### Task-Robust Pre-Training for Worst-Case Downstream Adaptation

**Authors:** Jianghui Wang, Yang Chen, Xingyu Xie, Cong Fang, Zhouchen Lin

### Task-aware world model learning with meta weighting via bi-level optimization

**Authors:** Huining Yuan, Hongkun Dou, Xingyu Jiang, Yue Deng

### TaskMet: Task-driven Metric Learning for Model Learning

**Authors:** Dishank Bansal, Ricky T. Q. Chen, Mustafa Mukadam, Brandon Amos

### Template-free Articulated Neural Point Clouds for Reposable View Synthesis

**Authors:** Lukas Uzolas, Elmar Eisemann, Petr Kellnhofer

### Temporal Robustness against Data poisoning

**Authors:** Wenxiao Wang, Soheil Feizi

### Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples

**Authors:** Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, He He

### The Gain from Ordering in Online Learning

**Authors:** Vasilis Kontonis, Mingchen Ma, Christos Tzamos

### The Impact of Positional Encoding on Length Generalization in Transformers

**Authors:** Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy

### The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit

**Authors:** Lorenzo Noci, Chuning Li, Mufan Li, Bobby He, Thomas Hofmann, Chris Maddison, Dan Roy

### The Transient Nature of Emergent In-Context Learning in Transformers

**Authors:** Aaditya Singh, Stephanie Chan, Ted Moskovitz, Erin Grant, Andrew Saxe, Felix Hill

### Three Towers: Flexible Contrastive Learning with Pretrained Image Models

**Authors:** Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Effrosyni Kokiopoulou

### Time-Independent Information-Theoretic Generalization Bounds for SGLD

**Authors:** Futoshi Futami, Masahiro Fujisawa

### [Spotlight] Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics

**Authors:** Leon Klein, Andrew Foong, Tor Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noe, Ryota Tomioka

### Token-Scaled Logit Distillation for Ternary Weight Generative Language Models

**Authors:** Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, Jungwook Choi

### [Oral] Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom

**Oral Presentation:** We, Dec 13, 08:15 -- Oral 3B

### [Oral] ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings

**Authors:** Shibo Hao, Tianyang Liu, Zhen Wang, Zhiting Hu

**Oral Presentation:** We, Dec 13, 08:00 -- Oral 3B

### Toward Better PAC-Bayes Bounds for Uniformly Stable Algorithms

**Authors:** Sijia Zhou, Yunwen Lei, Ata Kaban

### Towards Characterizing the First-order Query Complexity of Learning (Approximate) Nash Equilibria in Zero-sum Matrix Games

**Authors:** Hedi Hadiji, Sarah Sachs, Tim van Erven, Wouter Koolen

### Towards Free Data Selection with General-Purpose Models

**Authors:** Yichen Xie, Mingyu Ding, Masayoshi TOMIZUKA, Wei Zhan

### Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation

**Authors:** Haonan Wang, Xiaomeng Li

### Towards Label Position Bias in Graph Neural Networks

**Authors:** Haoyu Han, Xiaorui Liu, Feng Shi, MohamadAli Torkamani, Charu Aggarwal, Jiliang Tang

### Towards Optimal Effective Resistance Estimation

**Authors:** Rajat Vadiraj Dwaraknath, Ishani Karmarkar, Aaron Sidford

### [Spotlight] Towards Symmetry-Aware Generation of Periodic Materials

**Authors:** Youzhi Luo, Chengkai Liu, Shuiwang Ji

### Towards a Unified Framework of Contrastive Learning for Disentangled Representations

**Authors:** Stefan Matthes, Zhiwei Han, Hao Shen

### Trade-off Between Efficiency and Consistency for Removal-based Explanations

**Authors:** Yifan Zhang, Haowei He, Zhiquan Tan, Yang Yuan

### Trading-off price for data quality to achieve fair online allocation

**Authors:** Mathieu Molina, Nicolas Gast, Patrick Loiseau, Vianney Perchet

### Training Fully Connected Neural Networks is $\exists\mathbb{R}$-Complete

**Authors:** Daniel Bertschinger, Christoph Hertrich, Paul Jungeblut, Tillmann Miltzow, Simon Weber

### Training Private Models That Know What They Don’t Know

**Authors:** Stephan Rabanser, Anvith Thudi, Abhradeep Guha Thakurta, Krishnamurthy Dvijotham, Nicolas Papernot

### Training Transformers with 4-bit Integers

**Authors:** Haocheng Xi, ChangHao Li, Jianfei Chen, Jun Zhu

### Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies

**Authors:** Wayne Soo, Vishwa Goudar, Xiao-Jing Wang

### Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory

**Authors:** Minhak Song, Chulhee Yun

### Transfer Learning with Affine Model Transformation

**Authors:** Shunya Minami, Kenji Fukumizu, Yoshihiro Hayashi, Ryo Yoshida

### Transfer learning for atomistic simulations using GNNs and kernel mean embeddings

**Authors:** John Falk, Luigi Bonati, Pietro Novelli, Michele Parrinello, Massimiliano Pontil

### Transferable Adversarial Robustness for Categorical Data via Universal Robust Embeddings

**Authors:** Klim Kireev, Maksym Andriushchenko, Carmela Troncoso, Nicolas Flammarion

### Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars

**Authors:** Kaiyue Wen, Yuchen Li, Bingbin Liu, Andrej Risteski

### Transformers learn to implement preconditioned gradient descent for in-context learning

**Authors:** Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra

### [Spotlight] Transition-constant Normalization for Image Enhancement

**Authors:** Jie Huang, man zhou, Jinghao Zhang, Gang Yang, Mingde Yao, Chongyi Li, Zhiwei Xiong, Feng Zhao

### [Spotlight] Tree-Based Diffusion Schrödinger Bridge with Applications to Wasserstein Barycenters

**Authors:** Maxence Noble, Valentin De Bortoli, Arnaud Doucet, Alain Durmus

### Trial matching: capturing variability with data-constrained spiking neural networks

**Authors:** Christos Sourmpis, Carl Petersen, Wulfram Gerstner, Guillaume Bellec

### Truly Scale-Equivariant Deep Nets with Fourier Layers

**Authors:** Md Ashiqur Rahman, Raymond A. Yeh

### Truncating Trajectories in Monte Carlo Policy Evaluation: an Adaptive Approach

**Authors:** Riccardo Poiani, Nicole Nobili, Alberto Maria Metelli, Marcello Restelli

### Two-Stage Learning to Defer with Multiple Experts

**Authors:** Anqi Mao, Christopher Mohri, Mehryar Mohri, Yutao Zhong

### Type-to-Track: Retrieve Any Object via Prompt-based Tracking

**Authors:** Pha Nguyen, Kha Gia Quach, Kris Kitani, Khoa Luu

### Unbounded Differentially Private Quantile and Maximum Estimation

**Authors:** David Durfee

### Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation

**Authors:** Fei Zhang, Tianfei Zhou, Boyang Li, Hao He, Chaofan Ma, Tianjiao Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang

### [Oral] Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation

**Authors:** Diederik Kingma, Ruiqi Gao

**Oral Presentation:** We, Dec 13, 08:00 -- Oral 3C

### Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes

**Authors:** Minyang Hu, Hong Chang, Zong Guo, Bingpeng MA, Shiguang Shan, Xilin Chen

### Understanding and Improving Ensemble Adversarial Defense

**Authors:** Yian Deng, Tingting Mu

### Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions

**Authors:** Jun Xia, Lecheng Zhang, Xiao Zhu, Yue Liu, Zhangyang Gao, Bozhen Hu, Cheng Tan, Jiangbin Zheng, Siyuan Li, Stan Z. Li

### Unified 3D Segmenter As Prototypical Classifiers

**Authors:** Zheyun Qin, Cheng Han, Qifan Wang, Xiushan Nie, Yilong Yin, Lu Xiankai

### Unifying GANs and Score-Based Diffusion as Generative Particle Models

**Authors:** Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickael Chen, Alain Rakotomamonjy

### Universality laws for Gaussian mixtures in generalized linear models

**Authors:** Yatin Dandi, Ludovic Stephan, Florent Krzakala, Bruno Loureiro, Lenka Zdeborová

### Unlocking Feature Visualization for Deep Network with MAgnitude Constrained Optimization

**Authors:** Thomas FEL, Thibaut Boissin, Victor Boutin, Agustin PICARD, Paul Novello, Julien Colin, Drew Linsley, Tom ROUSSEAU, Remi Cadene, Lore Goetschalckx, Laurent Gardes, Thomas Serre

### Utilitarian Algorithm Configuration

**Authors:** Devon Graham, Kevin Leyton-Brown, Tim Roughgarden

### VCC: Scaling Transformers to 128K Tokens or More by Prioritizing Important Tokens

**Authors:** Zhanpeng Zeng, Cole Hawkins, Mingyi Hong, Aston Zhang, Nikolaos Pappas, Vikas Singh, Shuai Zheng

### VanillaNet: the Power of Minimalism in Deep Learning

**Authors:** Hanting Chen, Yunhe Wang, Jianyuan Guo, Dacheng Tao

### Variational Gaussian processes for linear inverse problems

**Authors:** Thibault RANDRIANARISOA, Botond Szabo

### Variational Monte Carlo on a Budget — Fine-tuning pre-trained Neural Wavefunctions

**Authors:** Michael Scherbela, Leon Gerard, Philipp Grohs

### Variational Weighting for Kernel Density Ratios

**Authors:** Sangwoong Yoon, Frank Park, Gunsu YUN, Iljung Kim, Yung-Kyun Noh

### VeriX: Towards Verified Explainability of Deep Neural Networks

**Authors:** Min Wu, Haoze Wu, Clark Barrett

### ViCA-NeRF: View-Consistency-Aware 3D Editing of Neural Radiance Fields

**Authors:** Jiahua Dong, Yu-Xiong Wang

### Video Prediction Models as Rewards for Reinforcement Learning

**Authors:** Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, Pieter Abbeel

### Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution

**Authors:** Ying Wang, Tim G. J. Rudner, Andrew Wilson

### Vocabulary-free Image Classification

**Authors:** Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, Elisa Ricci

### [Spotlight] VoxDet: Voxel Learning for Novel Instance Detection

**Authors:** Bowen Li, Jiashun Wang, Yaoyu Hu, Chen Wang, Sebastian Scherer

### Wasserstein distributional robustness of neural networks

**Authors:** Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

### Weakly Coupled Deep Q-Networks

**Authors:** Ibrahim El Shar, Daniel Jiang

### Weakly Supervised 3D Open-vocabulary Segmentation

**Authors:** Kunhao Liu, Fangneng Zhan, Jiahui Zhang, MUYU XU, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, Shijian Lu

### Weakly-Supervised Concealed Object Segmentation with SAM-based Pseudo Labeling and Multi-scale Feature Grouping

**Authors:** Chunming He, Kai Li, Yachao Zhang, Guoxia Xu, Longxiang Tang, Yulun Zhang, Zhenhua Guo, Xiu Li

### What Knowledge Gets Distilled in Knowledge Distillation?

**Authors:** Utkarsh Ojha, Yuheng Li, Anirudh Sundara Rajan, Yingyu Liang, Yong Jae Lee

### What functions can Graph Neural Networks compute on random graphs? The role of Positional Encoding

**Authors:** Nicolas Keriven, Samuel Vaiter

### When Can We Track Significant Preference Shifts in Dueling Bandits?

**Authors:** Joe Suk, Arpit Agarwal

### Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects

**Authors:** Chuanruo Ning, Ruihai Wu, Haoran Lu, Kaichun Mo, Hao Dong

### Why Did This Model Forecast This Future? Information-Theoretic Saliency for Counterfactual Explanations of Probabilistic Regression Models

**Authors:** Chirag Raman, Alec Nonnemaker, Amelia Villegas-Morcillo, Hayley Hung, Marco Loog

### Window-Based Distribution Shift Detection for Deep Neural Networks

**Authors:** Guy Bar Shalom, Yonatan Geifman, Ran El-Yaniv

### Zero-One Laws of Graph Neural Networks

**Authors:** Sam Adam-Day, Iliant, Ismail Ceylan

### Zero-Regret Performative Prediction Under Inequality Constraints

**Authors:** Wenjing YAN, Xuanyu Cao

### Zero-sum Polymatrix Markov Games: Equilibrium Collapse and Efficient Computation of Nash Equilibria

**Authors:** Fivos Kalogiannis, Ioannis Panageas

### [Spotlight] ZoomTrack: Target-aware Non-uniform Resizing for Efficient Visual Tracking

**Authors:** Yutong Kou, Jin Gao, Bing Li, Gang Wang, Weiming Hu, Yizheng Wang, Liang Li

### f-Policy Gradients: A General Framework for Goal-Conditioned RL using f-Divergences

**Authors:** Siddhant Agarwal, Ishan Durugkar, Peter Stone, Amy Zhang

### iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models

**Authors:** Tianyu Chen, Kevin Bello, Bryon Aragam, Pradeep Ravikumar

### “Why Not Looking backward?” A Robust Two-Step Method to Automatically Terminate Bayesian Optimization

**Authors:** Shuang Li, Ke Li, Wei Li

</details>

<details><summary><h3 style='display: inline;'> Poster Session 4: Wednesday, Dec 13, 15:00 CT</h3></summary>

### $\textbf{A}^2\textbf{CiD}^2$: Accelerating Asynchronous Communication in Decentralized Deep Learning

**Authors:** Adel Nabli, Eugene Belilovsky, Edouard Oyallon

### $k$-Means Clustering with Distance-Based Privacy

**Authors:** Alessandro Epasto, Vahab Mirrokni, Shyam Narayanan, Peilin Zhong

### $p$-Poisson surface reconstruction in curl-free flow from point clouds

**Authors:** Yesom Park, Taekyung Lee, Jooyoung Hahn, Myungjoo Kang

### 3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection

**Authors:** Yunhao Ge, Hong-Xing Yu, Cheng Zhao, Yuliang Guo, Xinyu Huang, Liu Ren, Laurent Itti, Jiajun Wu

### 3D molecule generation by denoising voxel grids

**Authors:** Pedro O. Pinheiro, Joshua Rackers, Joseph Kleinhenz, Michael Maser, Omar Mahmood, Andrew Watkins, Stephen Ra, Vishnu Sresht, Saeed Saremi

### [Spotlight] 3D-LLM: Injecting the 3D World into Large Language Models

**Authors:** Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, Chuang Gan

### [Spotlight] 4M: Massively Multimodal Masked Modeling

**Authors:** David Mizrahi, Roman Bachmann, Oguzhan Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, Amir Zamir

### A Bayesian Approach To Analysing Training Data Attribution In Deep Learning

**Authors:** Elisa Nguyen, Minjoon Seo, Seong Joon Oh

### A Bayesian Take on Gaussian Process Networks

**Authors:** Enrico Giudice, Jack Kuipers, Giusi Moffa

### A Definition of Continual Reinforcement Learning

**Authors:** David Abel, Andre Barreto, Benjamin Van Roy, Doina Precup, Hado van Hasselt, Satinder Singh

### [Spotlight] A Dynamical System View of Langevin-Based Non-Convex Sampling

**Authors:** Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause

### [Spotlight] A Graph-Theoretic Framework for Understanding Open-World Semi-Supervised Learning

**Authors:** Yiyou Sun, Zhenmei Shi, Yixuan Li

### A Heavy-Tailed Algebra for Probabilistic Programming

**Authors:** Feynman Liang, Liam Hodgkinson, Michael Mahoney

### A Hierarchical Spatial Transformer for Massive Point Samples  in Continuous Space

**Authors:** Wenchong He, Zhe Jiang, Tingsong Xiao, Zelin Xu, Shigang Chen, Ronald Fick, MILES MEDINA, Christine Angelini

### A Partially-Supervised Reinforcement Learning Framework for Visual Active Search

**Authors:** Anindya Sarkar, Nathan Jacobs, Yevgeniy Vorobeychik

### A Randomized Approach to Tight Privacy Accounting

**Authors:** Jiachen (Tianhao) Wang, Saeed Mahloujifar, Tong Wu, Ruoxi Jia, Prateek Mittal

### A Robust Exact Algorithm for the Euclidean Bipartite Matching Problem

**Authors:** Akshaykumar Gattani, Sharath Raghvendra, Pouyan Shirzadian

### [Spotlight] A Scalable Neural Network for DSIC Affine Maximizer Auction Design

**Authors:** Zhijian Duan, Haoran Sun, Yurong Chen, Xiaotie Deng

### [Oral] A Single-Loop Accelerated Extra-Gradient Difference Algorithm with Improved Complexity Bounds for Constrained Minimax Optimization

**Authors:** Yuanyuan Liu, Fanhua Shang, Weixin An, Junhao Liu, Hongying Liu, Zhouchen Lin

**Oral Presentation:** We, Dec 13, 13:30 -- Oral 4A

### A Unified Algorithm Framework for Unsupervised Discovery of Skills based on Determinantal Point Process

**Authors:** Jiayu Chen, Vaneet Aggarwal, Tian Lan

### A Unified Conditional Framework for Diffusion-based Image Restoration

**Authors:** Yi Zhang, Xiaoyu Shi, Dasong Li, Xiaogang Wang, Jian Wang, Hongsheng Li

### A Unified Framework for Rank-based Loss Minimization

**Authors:** Rufeng Xiao, Yuze Ge, Rujun Jiang, Yifan Yan

### [Spotlight] A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

**Authors:** Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang

### A Variational Perspective on High-Resolution ODEs

**Authors:** Hoomaan Maskan, Konstantinos Zygalakis, Alp Yurtsever

### A case for reframing automated medical image classification as segmentation

**Authors:** Sarah Hooper, Mayee Chen, Khaled Saab, Kush Bhatia, Curtis Langlotz, Christopher Ré

### A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs

**Authors:** Zhaocheng Zhu, Xinyu Yuan, Michael Galkin, Louis-Pascal Xhonneux, Ming Zhang, Maxime Gazeau, Jian Tang

### A-NeSI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference

**Authors:** Emile van Krieken, Thiviyan Thanapalasingam, Jakub Tomczak, Frank van Harmelen, Annette Ten Teije

### A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning

**Authors:** Hangfan Zhang, Jinyuan Jia, Jinghui Chen, Lu Lin, Dinghao Wu

### ANPL: Towards Natural Programming with Interactive Decomposition

**Authors:** Di Huang, Ziyuan Nan, Xing Hu, Pengwei Jin, Shaohui Peng, Yuanbo Wen, Rui Zhang, Zidong Du, Qi Guo, Yewen Pu, Yunji Chen

### [Spotlight] ARTree: A Deep Autoregressive Model for Phylogenetic Inference

**Authors:** Tianyu Xie, Cheng Zhang

### ASIF: Coupled Data Turns Unimodal Models to Multimodal without Training

**Authors:** Antonio Norelli, Marco Fumero, Valentino Maiorca, Luca Moschella, Emanuele Rodolà, Francesco Locatello

### ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation

**Authors:** Zhitong Gao, Shipeng Yan, Xuming He

### AVIDa-hIL6: A Large-Scale VHH Dataset Produced from an Immunized Alpaca for Predicting Antigen-Antibody Interactions

**Authors:** Hirofumi Tsuruta, Hiroyuki Yamazaki, Ryota Maeda, Ryotaro Tamura, Jennifer Wei, Zelda Mariet, Poomarin Phloyphisut, Hidetoshi Shimokawa, Joseph R. Ledsam, Lucy Colwell, Akihiro Imura

### Accelerating Value Iteration with Anchoring

**Authors:** Jongmin Lee, Ernest Ryu

### Accessing Higher Dimensions for Unsupervised Word Translation

**Authors:** Sida Wang

### Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models

**Authors:** Xingyuan Zhang, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl

### Active Reasoning in an Open-World Environment

**Authors:** Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu

### Active representation learning for general task space with applications in robotics

**Authors:** Yifang Chen, Yingbing Huang, Simon Du, Kevin Jamieson, Guanya Shi

### Actively Testing Your Model While It Learns: Realizing Label-Efficient Learning in Practice

**Authors:** Dayou Yu, Weishi Shi, Qi Yu

### AdANNS: A Framework for Adaptive Semantic Search

**Authors:** Aniket Rege, Aditya Kusupati, Sharan Ranjit S, Alan Fan, Qingqing Cao, Sham Kakade, Prateek Jain, Ali Farhadi

### Adapting to Continuous Covariate Shift via Online Density Ratio Estimation

**Authors:** Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, Masashi Sugiyama

### Adaptive SGD with Polyak stepsize and Line-search: Robust Convergence and Variance Reduction

**Authors:** Xiaowen Jiang, Sebastian Stich

### Adversarial Robustness through Random Weight Sampling

**Authors:** Yanxiang Ma, Minjing Dong, Chang Xu

### Agnostic Multi-Group Active Learning

**Authors:** Nicholas Rittler, Kamalika Chaudhuri

### Aiming towards the minimizers: fast convergence of SGD for overparametrized problems

**Authors:** Chaoyue Liu, Dmitriy Drusvyatskiy, Misha Belkin, Damek Davis, Yian Ma

### [Spotlight] Aleatoric and Epistemic Discrimination: Fundamental Limits of Fairness Interventions

**Authors:** Hao Wang, Luxi He, Rui Gao, Flavio Calmon

### An $\varepsilon$-Best-Arm Identification Algorithm for Fixed-Confidence and Beyond

**Authors:** Marc Jourdan, Rémy Degenne, Emilie Kaufmann

### An Efficient Doubly-Robust Test for the Kernel Treatment Effect

**Authors:** Diego Martinez Taboada, Aaditya Ramdas, Edward Kennedy

### An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations

**Authors:** Haoran Yang, Xiangyu Zhao, Yicong Li, Hongxu Chen, Guandong Xu

### An Inductive Bias for Tabular Deep Learning

**Authors:** Ege Beyazit, Jonathan Kozaczuk, Bo Li, Vanessa Wallace, Bilal Fadlallah

### An Optimal Structured Zeroth-order Algorithm for Non-smooth Optimization

**Authors:** Marco Rando, Cesare Molinari, Lorenzo Rosasco, Silvia Villa

### Anchor Data Augmentation

**Authors:** Nora Schneider, Shirin Goshtasbpour, Fernando Perez-Cruz

### [Spotlight] Anonymous and Copy-Robust Delegations for Liquid Democracy

**Authors:** Markus Utke, Ulrike Schmidt-Kraepelin

### Anytime Model Selection in Linear Bandits

**Authors:** Parnian Kassraie, Nicolas Emmenegger, Andreas Krause, Aldo Pacchiano

### [Spotlight] Approximate Heavy Tails in Offline (Multi-Pass) Stochastic Gradient Descent

**Authors:** Kruno Lehman, Alain Durmus, Umut Simsekli

### Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow

**Authors:** Sizhe Wei, Yuxi Wei, Yue Hu, Yifan Lu, Yiqi Zhong, Siheng Chen, Ya Zhang

### Attacks on Online Learners: a Teacher-Student Analysis

**Authors:** Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

### [Spotlight] Auditing for Human Expertise

**Authors:** Rohan Alur, Loren Laine, Darrick Li, Manish Raghavan, Devavrat Shah, Dennis Shung

### Autodecoding Latent 3D Diffusion Models

**Authors:** Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc V Gool, Sergey Tulyakov

### Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger

**Authors:** Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, George Karypis

### Autonomous Capability Assessment of Sequential Decision-Making Systems in Stochastic Settings

**Authors:** Pulkit Verma, Rushang Karia, Siddharth Srivastava

### [Oral] BEDD: The MineRL BASALT Evaluation and Demonstrations Dataset for Training and Benchmarking Agents that Solve Fuzzy Tasks

**Authors:** Stephanie Milani, Anssi Kanervisto, Karolis Ramanauskas, Sander Schulhoff, Brandon Houghton, Rohin Shah

**Oral Presentation:** We, Dec 13, 14:15 -- Oral 4B

### BERT Lost Patience Won't Be Robust to Adversarial Slowdown

**Authors:** Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong

### BIOT: Biosignal Transformer for Cross-data Learning in the Wild

**Authors:** Chaoqi Yang, M Westover, Jimeng Sun

### Back-Modality: Leveraging Modal Transformation for Data Augmentation

**Authors:** Zhi Li, Yifan Liu, Yin Zhang

### BadTrack: A Poison-Only Backdoor Attack on Visual Object Tracking

**Authors:** Bin Huang, Jiaqian Yu, Yiwei Chen, Siyang Pan, Qiang Wang, Zhi Wang

### Bandit Social Learning under Myopic Behavior

**Authors:** Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Aleksandrs Slivkins

### BanditPAM++: Faster $k$-medoids Clustering

**Authors:** Mo Tiwari, Ryan Kang, Donghyun Lee, Sebastian Thrun, Ilan Shomorony, Martin Zhang

### Batch Bayesian Optimization For Replicable Experimental Design

**Authors:** Zhongxiang Dai, Quoc Phong Nguyen, Sebastian Tay, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low, Patrick Jaillet

### Batchnorm Allows Unsupervised Radial Attacks

**Authors:** Amur Ghose, Apurv Gupta, Yaoliang Yu, Pascal Poupart

### BayesTune: Bayesian Sparse Deep Model Fine-tuning

**Authors:** Minyoung Kim, Timothy Hospedales

### [Spotlight] Bayesian Extensive-Rank Matrix Factorization with Rotational Invariant Priors

**Authors:** Farzad Pourkamali, Nicolas Macris

### Bayesian Optimisation of Functions on Graphs

**Authors:** Xingchen Wan, Pierre Osselin, Henry Kenlay, Binxin Ru, Michael A Osborne, Xiaowen Dong

### Bayesian Risk-Averse Q-Learning with Streaming Observations

**Authors:** Yuhao Wang, Enlu Zhou

### Beyond Black-Box Advice: Learning-Augmented Algorithms for MDPs with Q-Value Predictions

**Authors:** Tongxin Li, Yiheng Lin, Shaolei Ren, Adam Wierman

### Beyond Confidence: Reliable Models Should Also Consider Atypicality

**Authors:** Mert Yuksekgonul, Linjun Zhang, James Zou, Carlos Guestrin

### [Spotlight] Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends

**Authors:** Wang Xinrui, Wenhai Wan, Chuanxing Geng, Shao-Yuan Li, Songcan Chen

### Beyond NTK with Vanilla Gradient Descent: A Mean-Field Analysis of Neural Networks with Polynomial Width, Samples, and Time

**Authors:** Arvind Mahankali, Jeff Z. HaoChen, Kefan Dong, Margalit Glasgow, Tengyu Ma

### Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets

**Authors:** Zhang-Wei Hong, Aviral Kumar, Sathwik Karnik, Abhishek Bhandwaldar, Akash Srivastava, Joni Pajarinen, Romain Laroche, Abhishek Gupta, Pulkit Agrawal

### BiMatting: Efficient Video Matting via Binarization

**Authors:** Haotong Qin, Lei Ke, Xudong Ma, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Xianglong Liu, Fisher Yu

### Binary Radiance Fields

**Authors:** Seungjoo Shin, Jaesik Park

### Block-State Transformers

**Authors:** Jonathan Pilault, Mahan Fathi, Orhan Firat, Chris Pal, Pierre-Luc Bacon, Ross Goroshin

### Boosting Learning for LDPC Codes to Improve the Error-Floor Performance

**Authors:** Hee-Youl Kwak, Dae-Young Yun, Yongjune Kim, Sang-Hyo Kim, Jong-Seon No

### Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces

**Authors:** Leonard Papenmeier, Luigi Nardi, Matthias Poloczek

### Brain encoding models based on multimodal transformers can transfer across language and vision

**Authors:** Jerry Tang, Meng Du, Vy Vo, VASUDEV LAL, Alexander Huth

### Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs

**Authors:** Jacob Lindbäck, Zesen Wang, Mikael Johansson

### Bypass Exponential Time Preprocessing: Fast Neural Network Training via Weight-Data Correlation Preprocessing

**Authors:** Josh Alman, 杰昊 梁, Zhao Song, Ruizhe Zhang, Danyang Zhuo

### CELLE-2: Translating Proteins to Pictures and Back with a Bidirectional Text-to-Image Transformer

**Authors:** Emaad Khwaja, Yun Song, Aaron Agarunov, Bo Huang

### [Spotlight] CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions

**Authors:** Juan M. Cardenas, Ben Adcock, Nick Dexter

### [Spotlight] Can Language Models Solve Graph Problems in Natural Language?

**Authors:** Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov

### Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer

**Authors:** Bowen Tan, Yun Zhu, Lijuan Liu, Eric Xing, Zhiting Hu, Jindong Chen

### Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions

**Authors:** Kai Liu, Zhihang Fu, Chao Chen, Sheng Jin, Ze Chen, Mingyuan Tao, Rongxin Jiang, Jieping Ye

### Causal Discovery in Semi-Stationary Time Series

**Authors:** Shanyun Gao, Raghavendra Addanki, Tong Yu, Ryan Rossi, Murat Kocaoglu

### Causal Effect Regularization: Automated Detection and Removal of Spurious Correlations

**Authors:** Abhinav Kumar, Amit Deshpande, Amit Sharma

### Causal Imitability Under Context-Specific Independence Relations

**Authors:** Fateme Jamshidi, Sina Akbari, Negar Kiyavash

### Causal-structure Driven Augmentations for Text OOD Generalization

**Authors:** Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, David Blei

### Chanakya: Learning Runtime Decisions for Adaptive Real-Time Perception

**Authors:** Anurag Ghosh, Vaibhav Balloli, Akshay Nambi, Aditya Singh, Tanuja Ganu

### Characterization of Overfitting in Robust Multiclass Classification

**Authors:** Jingyuan Xu, Weiwei Liu

### Characterizing Out-of-Distribution Error via Optimal Transport

**Authors:** Yuzhe Lu, Yilong Qin, Runtian Zhai, Andrew Shen, Ketong Chen, Zhenlin Wang, Soheil Kolouri, Simon Stepputtis, Joseph Campbell, Katia Sycara

### Characterizing the Impacts of Semi-supervised Learning for Weak Supervision

**Authors:** Jeffrey Li, Jieyu Zhang, Ludwig Schmidt, Alexander Ratner

### Chatting Makes Perfect: Chat-based Image Retrieval

**Authors:** Matan Levy, Rami Ben-Ari, Nir Darshan, Dani Lischinski

### Class-Distribution-Aware Pseudo-Labeling for Semi-Supervised Multi-Label Learning

**Authors:** Ming-Kun Xie, Jiahao Xiao, Hao-Zhe Liu, Gang Niu, Masashi Sugiyama, Sheng-Jun Huang

### [Oral] ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation

**Authors:** Sungduk Yu, Walter Hannah, Liran Peng, Jerry Lin, Mohamed Aziz Bhouri, Ritwik Gupta, Björn Lütjens, Justus C. Will, Gunnar Behrens, Julius Busecke, Nora Loose, Charles Stern, Tom Beucler, Bryce Harrop, Benjamin Hillman, Andrea Jenney, Savannah L. Ferretti, Nana Liu, Animashree Anandkumar, Noah Brenowitz, Veronika Eyring, Nicholas Geneva, Pierre Gentine, Stephan Mandt, Jaideep Pathak, Akshay Subramaniam, Carl Vondrick, Rose Yu, Laure Zanna, Tian Zheng, Ryan Abernathey, Fiaz Ahmed, David Bader, Pierre Baldi, Elizabeth Barnes, Christopher Bretherton, Peter Caldwell, Wayne Chuang, Yilun Han, YU HUANG, Fernando Iglesias-Suarez, Sanket Jantre, Karthik Kashinath, Marat Khairoutdinov, Thorsten Kurth, Nicholas Lutsko, Po-Lun Ma, Griffin Mooers, J. David Neelin, David Randall, Sara Shamekh, Mark Taylor, Nathan Urban, Janni Yuval, Guang Zhang, Mike Pritchard

**Oral Presentation:** We, Dec 13, 13:45 -- Oral 4B

### Cluster-aware Semi-supervised Learning: Relational Knowledge Distillation Provably Learns Clustering

**Authors:** Yijun Dong, Kevin Miller, Qi Lei, Rachel Ward

### ClusterFomer: Clustering As A Universal Visual Learner

**Authors:** James Liang, Yiming Cui, Qifan Wang, Tong Geng, Wenguan Wang, Dongfang Liu

### CoDet: Co-occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection

**Authors:** Chuofan Ma, Yi Jiang, Xin Wen, Zehuan Yuan, Xiaojuan Qi

### CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra

**Authors:** Andres Potapczynski, Marc Finzi, Geoff Pleiss, Andrew Wilson

### Cognitive Model Discovery via Disentangled RNNs

**Authors:** Kevin Miller, Maria Eckstein, Matt Botvinick, Zeb Kurth-Nelson

### Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise

**Authors:** Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein

### Collaborative Learning via Prediction Consensus

**Authors:** Dongyang Fan, Celestine Mendler-Dünner, Martin Jaggi

### Compositional Foundation Models for Hierarchical Planning

**Authors:** Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, Pulkit Agrawal

### Compositional Policy Learning in Stochastic Control Systems with Formal Guarantees

**Authors:** Đorđe Žikelić, Mathias Lechner, Abhinav Verma, Krishnendu Chatterjee, Thomas Henzinger

### Concept Distillation: Leveraging Human-Centered Explanations for Model Improvement

**Authors:** Avani Gupta, Saurabh Saini, P J Narayanan

### Conditional Score Guidance for Text-Driven Image-to-Image Translation

**Authors:** Hyunsoo Lee, Minsoo Kang, Bohyung Han

### Conformal Prediction Sets for Ordinal Classification

**Authors:** Prasenjit Dey, Srujana Merugu, Sivaramakrishnan R Kaveri

### Conformal Prediction for Uncertainty-Aware Planning with Diffusion Dynamics Model

**Authors:** Jiankai Sun, Yiqi Jiang, Jianing Qiu, Parth Nobel, Mykel J Kochenderfer, Mac Schwager

### Connecting Certified and Adversarial Training

**Authors:** Yuhao Mao, Mark Müller, Marc Fischer, Martin Vechev

### Connecting Pre-trained Language Model and Downstream Task via Properties of Representation

**Authors:** Chenwei Wu, Holden Lee, Rong Ge

### Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning

**Authors:** Jing Zhang, Chi Zhang, Wenjia Wang, Bingyi Jing

### Context Shift Reduction for Offline Meta-Reinforcement Learning

**Authors:** Yunkai Gao, Rui Zhang, Jiaming Guo, Fan Wu, Qi Yi, Shaohui Peng, Siming Lan, Ruizhi Chen, Zidong Du, Xing Hu, Qi Guo, Ling Li, Yunji Chen

### [Spotlight] Context-PIPs: Persistent Independent Particles Demands Context Features

**Authors:** Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yitong Dong, Yijin Li, Hongsheng Li

### Context-guided Embedding Adaptation for Effective Topic Modeling in Low-Resource Regimes

**Authors:** Yishi Xu, Jianqiao Sun, Yudi Su, Xinyang Liu, Zhibin Duan, Bo Chen, Mingyuan Zhou

### Contextual Stochastic Bilevel Optimization

**Authors:** Yifan Hu, Jie Wang, Yao Xie, Andreas Krause, Daniel Kuhn

### Continuous Parametric Optical Flow

**Authors:** Jianqin Luo, Zhexiong Wan, yuxin mao, Bo Li, Yuchao Dai

### Contrast, Attend and Diffuse to Decode High-Resolution Images from Brain Activities

**Authors:** Jingyuan Sun, Mingxiao Li, Zijiao Chen, Yunhao Zhang, Shaonan Wang, Marie-Francine Moens

### [Spotlight] Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion

**Authors:** Yash Bhalgat, Iro Laina, João Henriques, Andrea Vedaldi, Andrew Zisserman

### Contrastive Sampling Chains in Diffusion Models

**Authors:** Junyu Zhang, Daochang Liu, Shichao Zhang, Chang Xu

### Convergence analysis of ODE models for accelerated first-order methods via positive semidefinite kernels

**Authors:** Jungbin Kim, Insoon Yang

### [Spotlight] Convergence of Alternating Gradient Descent for Matrix Factorization

**Authors:** Rachel Ward, Tamara Kolda

### Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems

**Authors:** Samuel Hurault, Ulugbek Kamilov, Arthur Leclaire, Nicolas Papadakis

### Cookie Consent Has Disparate Impact on Estimation Accuracy

**Authors:** Erik Miehling, Rahul Nair, Elizabeth Daly, Karthikeyan Natesan Ramamurthy, Robert Redmond

### Correlation Aware Sparsified Mean Estimation Using Random Projection

**Authors:** Shuli Jiang, PRANAY SHARMA, Gauri Joshi

### Correlative Information Maximization: A Biologically Plausible Approach to Supervised Deep Neural Networks without Weight Symmetry

**Authors:** Bariscan Bozkurt, Cengiz Pehlevan, Alper Erdogan

### CosNet: A Generalized Spectral Kernel Network

**Authors:** Yanfang Xue, Pengfei Fang, Jinyue Tian, Shipeng Zhu, hui xue

### Counterfactual Generation with Identifiability Guarantees

**Authors:** Hanqi Yan, Lingjing Kong, Lin Gui, Yuejie Chi, Eric Xing, Yulan He, Kun Zhang

### Counterfactual-Augmented Importance Sampling for Semi-Offline Policy Evaluation

**Authors:** Shengpu Tang, Jenna Wiens

### Cross-Episodic Curriculum for Transformer Agents

**Authors:** Lucy Xiaoyang Shi, Yunfan Jiang, Jake Grigsby, Linxi Fan, Yuke Zhu

### DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets

**Authors:** Yash Jain, Harkirat Behl, Zsolt Kira, Vibhav Vineet

### DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

**Authors:** Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu

### Data Minimization at Inference Time

**Authors:** Cuong Tran, Nando Fioretto

### Dataset Diffusion: Diffusion-based Synthetic Data Generation for Pixel-Level Semantic Segmentation

**Authors:** Quang Nguyen, Truong Vu, Anh Tran, Khoi Nguyen

### [Spotlight] DeWave: Discrete Encoding of EEG Waves for EEG to Text Translation

**Authors:** Yiqun Duan, Charles Chau, Zhen Wang, Yu-Kai Wang, Chin-teng Lin

### Decompose Novel into Known: Part Concept Learning For 3D Novel Class Discovery

**Authors:** Tingyu Weng, Jun Xiao, Haiyong Jiang

### Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning

**Authors:** Zikang Tian, Ruizhi Chen, Xing Hu, Ling Li, Rui Zhang, Fan Wu, Shaohui Peng, Jiaming Guo, Zidong Du, Qi Guo, Yunji Chen

### [Spotlight] Deep Fractional Fourier Transform

**Authors:** Hu Yu, Jie Huang, Lingzhi LI, man zhou, Feng Zhao

### Deep Gaussian Markov Random Fields for Graph-Structured Dynamical Systems

**Authors:** Fiona Lippert, Bart Kranstauber, Emiel van Loon, Patrick Forré

### Deep Insights into Noisy Pseudo Labeling on Graph Data

**Authors:** Botao WANG, Jia Li, Yang Liu, Jiashun Cheng, Yu Rong, Wenjia Wang, Fugee Tsung

### Deep Patch Visual Odometry

**Authors:** Zachary Teed, Lahav Lipson, Jia Deng

### Deep Recurrent Optimal Stopping

**Authors:** Niranjan Damera Venkata, Chiranjib Bhattacharyya

### DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization

**Authors:** Haoran Ye, Jiarui Wang, Zhiguang Cao, Helan Liang, Yong Li

### DeepPCR: Parallelizing Sequential Operations in Neural Networks

**Authors:** Federico Danieli, Miguel Sarabia, Xavier Suau Cuadros, Pau Rodriguez, Luca Zappella

### DeepSimHO: Stable Pose Estimation for Hand-Object Interaction via Physics Simulation

**Authors:** Rong Wang, Wei Mao, Hongdong Li

### [Spotlight] Delegated Classification

**Authors:** Eden Saig, Inbal Talgam-Cohen, Nir Rosenfeld

### [Spotlight] Demystifying Softmax Gating Function in Gaussian Mixture of Experts

**Authors:** Huy Nguyen, TrungTin Nguyen, Nhat Ho

### Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel

**Authors:** Valerii Likhosherstov, Krzysztof M Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamas Sarlos, Adrian Weller

### Depth-discriminative Metric Learning for Monocular 3D Object Detection

**Authors:** Wonhyeok Choi, Mingyu Shin, Sunghoon Im

### Detection Based Part-level Articulated Object Reconstruction from Single RGBD Image

**Authors:** Yuki Kawana, Tatsuya Harada

### DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation

**Authors:** Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, Zhenguo Li

### DiViNeT: 3D Reconstruction from Disparate Views using Neural Template Regularization

**Authors:** Aditya Vora, Akshay Gadi Patil, Hao Zhang

### DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation

**Authors:** Kaipeng Zheng, Huishuai Zhang, Weiran Huang

### DiffUTE: Universal Text Editing Diffusion Model

**Authors:** Haoxing Chen, Zhuoer Xu, Zhangxuan Gu, jun lan, 行 郑, Yaohui Li, Changhua Meng, Huijia Zhu, Weiqiang Wang

### [Spotlight] Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching

**Authors:** Junsheng Zhou, Baorui Ma, Wenyuan Zhang, Yi Fang, Yu-Shen Liu, Zhizhong Han

### Differentially Private Decoupled Graph Convolutions for Multigranular Topology Protection

**Authors:** Eli Chien, Wei-Ning Chen, Chao Pan, Pan Li, Ayfer Ozgur, Olgica Milenkovic

### Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence

**Authors:** Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, Trevor Darrell

### Diffusion Self-Guidance for Controllable Image Generation

**Authors:** Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, Aleksander Holynski

### Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning

**Authors:** Seungyong Moon, Junyoung Yeom, Bumsoo Park, Hyun Oh Song

### Discovering Intrinsic Spatial-Temporal Logic Rules to Explain Human Actions

**Authors:** Chengzhi Cao, Chao Yang, Ruimao Zhang, Shuang Li

### Distributed Inference and Fine-tuning of Large Language Models Over The Internet

**Authors:** Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, Colin Raffel

### Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

**Authors:** Tyler Kastner, Murat Erdogdu, Amir-massoud Farahmand

### Distributionally Robust Bayesian Optimization with $\varphi$-divergences

**Authors:** Hisham Husain, Vu Nguyen, Anton van den Hengel

### Distributionally Robust Ensemble of Lottery Tickets Towards Calibrated  Sparse Network Training

**Authors:** Hitesh Sapkota, Dingrong Wang, Zhiqiang Tao, Qi Yu

### Diversified Outlier Exposure for Out-of-Distribution Detection via Informative Extrapolation

**Authors:** Jianing Zhu, Yu Geng, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, Bo Han

### Diversifying Spatial-Temporal Perception for Video Domain Generalization

**Authors:** Kun-Yu Lin, Jia-Run Du, Yipeng Gao, Jiaming Zhou, Wei-Shi Zheng

### [Spotlight] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

**Authors:** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, Adams Wei Yu

### Does Graph Distillation See Like Vision Dataset Counterpart?

**Authors:** Beining Yang, Kai Wang, Qingyun Sun, Cheng Ji, Xingcheng Fu, Hao Tang, Yang You, Jianxin Li

### Does a sparse ReLU network training problem always admit an optimum ?

**Authors:** QUOC-TUNG LE, Remi Gribonval, Elisa Riccietti

### [Spotlight] Double Gumbel Q-Learning

**Authors:** David Yu-Tung Hui, Aaron Courville, Pierre-Luc Bacon

### Double Pessimism is Provably Efficient for Distributionally Robust Offline Reinforcement Learning: Generic Algorithm and Robust Partial Coverage

**Authors:** Jose Blanchet, Miao Lu, Tong Zhang, Han Zhong

### Double and Single Descent in Causal Inference with an Application to High-Dimensional Synthetic Control

**Authors:** Jann Spiess, guido imbens, Amar Venugopal

### Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection

**Authors:** Chengsen Wang, Zirui Zhuang, Qi Qi, Jingyu Wang, Xingyu Wang, Haifeng Sun, Jianxin Liao

### DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions

**Authors:** Haochen Wang, Junsong Fan, Yuxi Wang, Kaiyou Song, Tong Wang, ZHAO-XIANG ZHANG

### DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets

**Authors:** Lazar Atanackovic, Alexander Tong, Bo Wang, Leo J Lee, Yoshua Bengio, Jason Hartford

### Dynamic Personalized Federated Learning with Adaptive Differential Privacy

**Authors:** Xiyuan Yang, Wenke Huang, Mang Ye

### Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes

**Authors:** Yihong Sun, Bharath Hariharan

### EDGI: Equivariant Diffusion for Planning with Embodied Agents

**Authors:** Johann Brehmer, Joey Bose, Pim de Haan, Taco Cohen

### EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning

**Authors:** Ping Guo, Xiangpeng Wei, Yue Hu, Baosong Yang, Dayiheng Liu, Fei Huang, jun xie

### Effective Bayesian Heteroscedastic Regression with Deep Neural Networks

**Authors:** Alexander Immer, Emanuele Palumbo, Alexander Marx, Julia Vogt

### [Spotlight] Effective Human-AI Teams via Learned Natural Language Rules and Onboarding

**Authors:** Hussein Mozannar, Jimin Lee, Dennis Wei, Prasanna Sattigeri, Subhro Das, David Sontag

### Efficient Activation Function Optimization through Surrogate Modeling

**Authors:** Garrett Bingham, Risto Miikkulainen

### Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing

**Authors:** Wei Dong, Dawei Yan, Zhijun Lin, Peng Wang

### Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards

**Authors:** Bo Xue, Yimu Wang, Yuanyu Wan, Jinfeng Yi, Lijun Zhang

### Efficient Equivariant Transfer Learning from Pretrained Models

**Authors:** Sourya Basu, Pulkit Katdare, Prasanna Sattigeri, Vijil Chenthamarakshan, Katherine Driggs-Campbell, Payel Das, Lav Varshney

### Efficient Exploration in Continuous-time Model-based Reinforcement Learning

**Authors:** Lenart Treven, Jonas Hübotter, Bhavya, Florian Dorfler, Andreas Krause

### Efficient RL with Impaired Observability: Learning to Act with Delayed and Missing State Observations

**Authors:** Minshuo Chen, Yu Bai, H. Vincent Poor, Mengdi Wang

### Efficiently incorporating quintuple interactions into geometric deep learning force fields

**Authors:** Zun Wang, Guoqing Liu, Yichi Zhou, Tong Wang, Bin Shao

### Elastic Decision Transformer

**Authors:** Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya

### Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss

**Authors:** An Zhang, Leheng Sheng, Zhibo Cai, Xiang Wang, Tat-Seng Chua

### Empowering Convolutional Neural Nets with MetaSin Activation

**Authors:** Farnood Salehi, Tunç Aydin, André Gaillard, Guglielmo Camporese, Yuxuan Wang

### Encoding Human Behavior in Information Design through Deep Learning

**Authors:** Guanghui Yu, Wei Tang, Saumik Narayanan, Chien-Ju Ho

### Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks

**Authors:** Qi Xu, Yuyuan Gao, Jiangrong Shen, Yaxin Li, Xuming Ran, Huajin Tang, Gang Pan

### Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams

**Authors:** Shiyan Chen, Jiyuan Zhang, Yajing Zheng, Tiejun Huang, Zhaofei Yu

### Enhancing Sharpness-Aware Optimization Through Variance Suppression

**Authors:** Bingcong Li, Georgios Giannakis

### [Spotlight] Epistemic Neural Networks

**Authors:** Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, MORTEZA IBRAHIMI, Xiuyuan Lu, Benjamin Van Roy

### Equal Opportunity of Coverage in Fair Regression

**Authors:** Fangxin Wang, Lu Cheng, Ruocheng Guo, Kay Liu, Philip S Yu

### Equivariant Single View Pose Prediction Via Induced and Restriction Representations

**Authors:** Owen Howell, David Klee, Ondrej Biza, Linfeng Zhao, Robin Walters

### Errors-in-variables Fr\'echet Regression with Low-rank Covariate Approximation

**Authors:** Dogyoon Song, Kyunghee Han

### Estimating Riemannian Metric with Noise-Contaminated Intrinsic Distance

**Authors:** Jiaming Qiu, Xiongtao Dai

### Estimating and Controlling for Equalized Odds via Sensitive Attribute Predictors

**Authors:** Beepul Bharti, Paul Yi, Jeremias Sulam

### Estimating the Rate-Distortion Function by Wasserstein Gradient Descent

**Authors:** Yibo Yang, Stephan Eckstein, Marcel Nutz, Stephan Mandt

### Evaluating Cognitive Maps and Planning in Large Language Models with CogEval

**Authors:** Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Ness, Jonathan Larson

### Evaluating Neuron Interpretation Methods of NLP Models

**Authors:** Yimin Fan, Fahim Dalvi, Nadir Durrani, Hassan Sajjad

### Exact Representation of Sparse Networks with Symmetric Nonnegative Embeddings

**Authors:** Sudhanshu Chanpuriya, Ryan Rossi, Anup Rao, Tung Mai, Nedim Lipka, Zhao Song, Cameron Musco

### Exact recovery and Bregman hard clustering of node-attributed Stochastic Block Model

**Authors:** Maximilien Dreveton, Felipe Fernandes, Daniel Figueiredo

### Experiment Planning with Function Approximation

**Authors:** Aldo Pacchiano, Jonathan Lee, Emma Brunskill

### Expert load matters: operating networks at high accuracy and low manual effort

**Authors:** Sara Sangalli, Ertunc Erdil, Ender Konukoglu

### Explain Any Concept: Segment Anything Meets Concept-Based Explanation

**Authors:** Ao Sun, Pingchuan Ma, Yuanyuan Yuan, Shuai Wang

### Explainable Brain Age Prediction using coVariance Neural Networks

**Authors:** Saurabh Sihag, Gonzalo Mateos, Corey McMillan, Alejandro Ribeiro

### Exploring Diverse In-Context Configurations for Image Captioning

**Authors:** Xu Yang, Yongliang Wu, Mingzhuo Yang, Haokun Chen, Xin Geng

### [Spotlight] Exposing Attention Glitches with Flip-Flop Language Modeling

**Authors:** Bingbin Liu, Jordan Ash, Surbhi Goel, Akshay Krishnamurthy, Cyril Zhang

### [Spotlight] Expressive Sign Equivariant Networks for Spectral Geometric Learning

**Authors:** Derek Lim, Joshua Robinson, Stefanie Jegelka, Haggai Maron

### Expressive probabilistic sampling in recurrent neural networks

**Authors:** Shirui Chen, Linxing Jiang, Rajesh PN Rao, Eric Shea-Brown

### Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman

**Authors:** Jiarui Feng, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, Yixin Chen

### Factorized Contrastive Learning: Going Beyond Multi-view Redundancy

**Authors:** Paul Pu Liang, Zihao Deng, Martin Q. Ma, James Zou, Louis-Philippe Morency, Ruslan Salakhutdinov

### Fair Canonical Correlation Analysis

**Authors:** Zhuoping Zhou, Davoud Ataee Tarzanagh, Bojian Hou, Boning Tong, Jia Xu, Yanbo Feng, Qi Long, Li Shen

### Fair, Polylog-Approximate Low-Cost Hierarchical Clustering

**Authors:** Marina Knittel, Max Springer, John Dickerson, MohammadTaghi Hajiaghayi

### Fast Attention Over Long Sequences With Dynamic Sparse Flash Attention

**Authors:** Matteo Pagliardini, Daniele Paliotta, Martin Jaggi, François Fleuret

### Fast Exact Leverage Score Sampling from Khatri-Rao Products with Applications to Tensor Decomposition

**Authors:** Vivek Bharadwaj, Osman Asif Malik, Riley Murray, Laura Grigori, Aydin Buluc, James Demmel

### Fast Trainable Projection for Robust Fine-tuning

**Authors:** Junjiao Tian, Yen-Cheng Liu, James S Smith, Zsolt Kira

### Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms

**Authors:** Qining Zhang, Lei Ying

### Feature Selection in the Contrastive Analysis Setting

**Authors:** Ethan Weinberger, Ian Covert, Su-In Lee

### Fed-CO$_{2}$: Cooperation of Online and Offline Models for Severe Data Heterogeneity in Federated Learning

**Authors:** Zhongyi Cai, Ye Shi, Wei Huang, Jingya Wang

### Federated Learning via Meta-Variational Dropout

**Authors:** Insu Jeon, Minui Hong, Junhyeog Yun, Gunhee Kim

### Federated Learning with Client Subsampling, Data Heterogeneity, and Unbounded Smoothness: A New Algorithm and Lower Bounds

**Authors:** Michael Crawshaw, Yajie Bao, Mingrui Liu

### Federated Linear Bandits with Finite Adversarial Actions

**Authors:** Li Fan, Ruida Zhou, Chao Tian, Cong Shen

### Federated Spectral Clustering via Secure Similarity Reconstruction

**Authors:** Dong Qiao, Chris Ding, Jicong Fan

### Few-shot Generation via Recalling  Brain-Inspired Episodic-Semantic Memory

**Authors:** Zhibin Duan, Zhiyi Lv, Chaojie Wang, Bo Chen, Bo An, Mingyuan Zhou

### Finding Counterfactually Optimal Action Sequences in Continuous State Spaces

**Authors:** Stratis Tsirtsis, Manuel Rodriguez

### Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning

**Authors:** Berken Utku Demirel, Christian Holz

### [Oral] Fine-Tuning Language Models with Just Forward Passes

**Authors:** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason Lee, Danqi Chen, Sanjeev Arora

**Oral Presentation:** We, Dec 13, 14:15 -- Oral 4A

### Fixing the NTK: From Neural Network Linearizations to Exact Convex Programs

**Authors:** Rajat Vadiraj Dwaraknath, Tolga Ergen, Mert Pilanci

### FlowCam: Training Generalizable 3D Radiance Fields without Camera Poses via Pixel-Aligned Scene Flow

**Authors:** Cameron Smith, Yilun Du, Ayush Tewari, Vincent Sitzmann

### FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

**Authors:** Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu

### FourierHandFlow: Neural 4D Hand Representation Using Fourier Query Flow

**Authors:** Jihyun Lee, Junbong Jang, Donghwan Kim, Minhyuk Sung, Tae-Kyun Kim

### From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion

**Authors:** Robin San Roman, Yossi Adi, Antoine Deleforge, Romain Serizel, Gabriel Synnaeve, Alexandre Defossez

### [Spotlight] Full-Atom Protein Pocket Design via Iterative Refinement

**Authors:** ZAIXI ZHANG, Zepu Lu, Hao Zhongkai, Marinka Zitnik, Qi Liu

### Fully Dynamic $k$-Clustering in $\tilde O(k)$ Update Time

**Authors:** Sayan Bhattacharya, Martín Costa, Silvio Lattanzi, Nikos Parotsidis

### Functional Equivalence and Path Connectivity of Reducible Hyperbolic Tangent Networks

**Authors:** Matthew Farrugia-Roberts

### GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference

**Authors:** Ziang Li, Mengda Yang, Yaxin Liu, Juan Wang, Hongxin Hu, Wenzhe Yi, Xiaoyang Xu

### GEQ: Gaussian Kernel Inspired Equilibrium Models

**Authors:** Mingjie Li, Yisen Wang, Zhouchen Lin

### GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning

**Authors:** Haiteng Zhao, Shengchao Liu, Ma Chang, Hannan Xu, Jie Fu, Zhihong Deng, Lingpeng Kong, Qi Liu

### GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER

**Authors:** Mingzhen Sun, Weining Wang, Zihan Qin, Jiahui Sun, Sihan Chen, Jing Liu

### GMSF: Global Matching Scene Flow

**Authors:** Yushan Zhang, Johan Edstedt, Bastian Wandt, Per-Erik Forssen, Maria Magnusson, Michael Felsberg

### GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction

**Authors:** Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan

### Gacs-Korner Common Information Variational Autoencoder

**Authors:** Michael Kleinman, Alessandro Achille, Stefano Soatto, Jonathan Kao

### Generalized Bayesian Inference for Scientific Simulators via Amortized Cost Estimation

**Authors:** Richard Gao, Michael Deistler, Jakob H Macke

### [Oral] Generalizing Nonlinear ICA Beyond Structural Sparsity

**Authors:** Yujia Zheng, Kun Zhang

**Oral Presentation:** We, Dec 13, 14:00 -- Oral 4A

### Graph Convolutional Kernel Machine versus Graph Convolutional Networks

**Authors:** Zhihao Wu, Zhao Zhang, Jicong Fan

### Graph of Circuits with GNN for Exploring the Optimal Design Space

**Authors:** Aditya Shahane, Saripilli Swapna Manjiri, Ankesh Jain, Sandeep Kumar

### GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph

**Authors:** Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang

### GraphMP: Graph Neural Network-based Motion Planning with Efficient Graph Search

**Authors:** Xiao Zang, Miao Yin, Jinqi Xiao, Saman Zonouz, Bo Yuan

### Greedy Poisson Rejection Sampling

**Authors:** Gergely Flamich

### [Spotlight] Group Fairness in Peer Review

**Authors:** Haris Aziz, Evi Micha, Nisarg Shah

### Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability

**Authors:** Revan MacQueen, James Wright

### Guiding Large Language Models via Directional Stimulus Prompting

**Authors:** Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan

### [Spotlight] HIQL: Offline Goal-Conditioned RL with Latent States as Actions

**Authors:** Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

### Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery

**Authors:** Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, Tom Goldstein

### Have it your way: Individualized Privacy Assignment for DP-SGD

**Authors:** Franziska Boenisch, Christopher Mühl, Adam Dziedzic, Roy Rinberg, Nicolas Papernot

### [Spotlight] Hierarchical Integration Diffusion Model for Realistic Image Deblurring

**Authors:** Zheng Chen, Yulun Zhang, Ding Liu, bin xia, Jinjin Gu, Linghe Kong, Xin Yuan

### Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection

**Authors:** Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, Ruimin Hu

### Horospherical Decision Boundaries for Large Margin Classification in Hyperbolic Space

**Authors:** Xiran Fan, Chun-Hao Yang, Baba Vemuri

### Hybrid Search for Efficient Planning with Completeness Guarantees

**Authors:** Kalle Kujanpää, Joni Pajarinen, Alexander Ilin

### Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach

**Authors:** Nurendra Choudhary, Nikhil Rao, Chandan Reddy

### Hyperbolic Space with Hierarchical Margin Boosts Fine-Grained Learning from Coarse Labels

**Authors:** Shu-Lin Xu, Yifan Sun, Faen Zhang, Anqi Xu, Xiu-Shen Wei, Yi Yang

### [Spotlight] Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks

**Authors:** Woojin Cho, Kookjin Lee, Donsub Rim, Noseong Park

### IBA: Towards Irreversible Backdoor Attacks in Federated Learning

**Authors:** Thuy Dung Nguyen, Tuan Nguyen, Anh Tran, Khoa D Doan, Kok-Seng Wong

### IMPRESS: Evaluating the Resilience of Imperceptible Perturbations Against Unauthorized Data Usage in Diffusion-Based  Generative AI

**Authors:** Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen

### Identification of Nonlinear Latent Hierarchical Models

**Authors:** Lingjing Kong, Biwei Huang, Feng Xie, Eric Xing, Yuejie Chi, Kun Zhang

### Implicit Manifold Gaussian Process Regression

**Authors:** Bernardo Fichera, Slava Borovitskiy, Andreas Krause, Aude G Billard

### Implicit Regularization in Over-Parameterized Support Vector Machine

**Authors:** Yang Sui, Xin HE, Yang Bai

### [Spotlight] Implicit Variational Inference for High-Dimensional Posteriors

**Authors:** Anshuk Uppal, Kristoffer Stensbo-Smidt, Wouter Boomsma, Jes Frellsen

### Improving Adversarial Transferability via Intermediate-level Perturbation Decay

**Authors:** Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

### Improving CLIP Training with Language Rewrites

**Authors:** Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, Yonglong Tian

### Improving Robustness with Adaptive Weight Decay

**Authors:** Mohammad Amin Ghiasi, Ali Shafahi, Reza Ardekani

### Incentives in Private Collaborative Machine Learning

**Authors:** Rachael Sim, Yehong Zhang, Nghia Hoang, Xinyi Xu, Bryan Kian Hsiang Low, Patrick Jaillet

### [Spotlight] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model

**Authors:** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg

### Injecting Multimodal Information into Rigid Protein Docking via Bi-level Optimization

**Authors:** Ruijia Wang, YiWu Sun, Yujie Luo, Shaochuan Li, Cheng Yang, Xingyi Cheng, Hui Li, Chuan Shi, Le Song

### Inner-Outer Aware Reconstruction Model for Monocular 3D Scene Reconstruction

**Authors:** Yu-Kun Qiu, Guo-Hao Xu, Wei-Shi Zheng

### Integration-free Training for Spatio-temporal Multimodal Covariate Deep Kernel Point Processes

**Authors:** YIXUAN ZHANG, Quyu Kong, Feng Zhou

### Interactive Multi-fidelity Learning for Cost-effective Adaptation of Language Model with Sparse Human Supervision

**Authors:** Jiaxin Zhang, Zhuohang Li, Kamalika Das, Sricharan Kumar

### Inverse Reinforcement Learning with the Average Reward Criterion

**Authors:** Feiyang Wu, Jingyang Ke, Anqi Wu

### Is This Loss Informative? Faster Text-to-Image Customization by Tracking Objective Dynamics

**Authors:** Anton Voronov, Mikhail Khoroshikh, Artem Babenko, Max Ryabinin

### Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization

**Authors:** Shurui Gui, Meng Liu, Xiner Li, Youzhi Luo, Shuiwang Ji

### Joint Training of Deep Ensembles Fails Due to Learner Collusion

**Authors:** Alan Jeffares, Tennison Liu, Jonathan Crabbé, Mihaela van der Schaar

### Keypoint-Augmented Self-Supervised Learning for Medical Image Segmentation with Limited Annotation

**Authors:** Zhangsihao Yang, Mengwei Ren, Kaize Ding, Guido Gerig, Yalin Wang

### [Spotlight] L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors

**Authors:** zheng chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi

### LEACE: Perfect linear concept erasure in closed form

**Authors:** Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman

### LICO: Explainable Models with Language-Image COnsistency

**Authors:** Yiming Lei, Zilong Li, Yangyang Li, Junping Zhang, Hongming Shan

### LMC: Large Model Collaboration with Cross-assessment for Training-Free Open-Set Object Recognition

**Authors:** Haoxuan Qu, Xiaofei Hui, Yujun Cai, Jun Liu

### Language Is Not All You Need: Aligning Perception with Language Models

**Authors:** Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Nils Bjorck, Vishrav Chaudhary, Subhojit Som, XIA SONG, Furu Wei

### Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning

**Authors:** Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei

### Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

**Authors:** Miles Turpin, Julian Michael, Ethan Perez, Samuel Bowman

### Language Models Meet World Models: Embodied Experiences Enhance Language Models

**Authors:** Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, Zhiting Hu

### Large Language Models Are Semi-Parametric Reinforcement Learning Agents

**Authors:** Danyang Zhang, Lu Chen, Situo Zhang, Hongshen Xu, Zihan Zhao, Kai Yu

### Large Language Models as Commonsense Knowledge for Large-Scale Task Planning

**Authors:** Zirui Zhao, Wee Sun Lee, David Hsu

### Large Language Models of Code Fail at Completing Code with Potential Bugs

**Authors:** Tuan Dinh, Jinman Zhao, Samson Tan, Renato Negrinho, Leonard Lausen, Sheng Zha, George Karypis

### Large-Scale Distributed Learning via Private On-Device LSH

**Authors:** Tahseen Rabbani, Marco Bornstein, Furong Huang

### Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs

**Authors:** Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, Alejandro Ribeiro

### Latent SDEs on Homogeneous Spaces

**Authors:** Sebastian Zeng, Florian Graf, Roland Kwitt

### LayoutGPT: Compositional Visual Planning and Generation with Large Language Models

**Authors:** Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Xuehai He, S Basu, Xin Eric Wang, William Yang Wang

### Learning Adversarial Low-rank Markov Decision Processes with Unknown Transition and Full-information Feedback

**Authors:** Canzhe Zhao, Ruofeng Yang, Baoxiang Wang, Xuezhou Zhang, Shuai Li

### Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning

**Authors:** Fan Feng, Sara Magliacane

### Learning Environment-Aware Affordance for 3D Articulated Object Manipulation under Occlusions

**Authors:** Ruihai Wu, Kai Cheng, Yan Zhao, Chuanruo Ning, Guanqi Zhan, Hao Dong

### [Spotlight] Learning Layer-wise Equivariances Automatically using Gradients

**Authors:** Tycho van der Ouderaa, Alexander Immer, Mark van der Wilk

### Learning Nonparametric Latent Causal Graphs with Unknown Interventions

**Authors:** Yibo Jiang, Bryon Aragam

### Learning Regularized Monotone Graphon Mean-Field Games

**Authors:** Fengzhuo Zhang, Vincent Tan, Zhaoran Wang, Zhuoran Yang

### Learning Robust Statistics for Simulation-based Inference under Model Misspecification

**Authors:** Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, Samuel Kaski

### Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction

**Authors:** Tianyu Liu, Qitan Lv, Jie Wang, Shuling Yang, Hanzhu Chen

### Learning Trajectories are Generalization Indicators

**Authors:** Jingwen Fu, Zhizheng Zhang, Dacheng Yin, Yan Lu, Nanning Zheng

### [Spotlight] Learning Universal Policies via Text-Guided Video Generation

**Authors:** Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, Pieter Abbeel

### Learning non-Markovian Decision-Making from State-only Sequences

**Authors:** Aoyang Qin, Feng Gao, Qing Li, Song-Chun Zhu, Sirui Xie

### Learning to Modulate pre-trained Models in RL

**Authors:** Thomas Schmied, Markus Hofmarcher, Fabian Paischer, Razvan Pascanu, Sepp Hochreiter

### Learning to Reason and Memorize with Self-Notes

**Authors:** Jack Lanchantin, Shubham Toshniwal, Jason Weston, arthur szlam, Sainbayar Sukhbaatar

### Learning to Tokenize for Generative Retrieval

**Authors:** Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten Rijke, Zhaochun Ren

### Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification

**Authors:** Rui Wang, Peipei Li, Huaibo Huang, Chunshui Cao, Ran He, Zhaofeng He

### Leveraging Locality and Robustness to Achieve Massively Scalable Gaussian Process Regression

**Authors:** Robert Allison, Anthony Stephenson, Samuel F, Edward O Pyzer-Knapp

### Lie Point Symmetry and Physics-Informed Networks

**Authors:** Tara Akhound-Sadegh, Laurence Perreault-Levasseur, Johannes Brandstetter, Max Welling, Siamak Ravanbakhsh

### LightSpeed: Light and Fast Neural Light Fields on Mobile Devices

**Authors:** Aarush Gupta, Junli Cao, Chaoyang Wang, Ju Hu, Sergey Tulyakov, Jian Ren, László Jeni

### Likelihood-Based Diffusion Language Models

**Authors:** Ishaan Gulrajani, Tatsunori Hashimoto

### Local Convergence of Gradient Methods for Min-Max Games: Partial Curvature Generically Suffices

**Authors:** Guillaume Wang, Lénaïc Chizat

### Localized Symbolic Knowledge Distillation for Visual Commonsense Models

**Authors:** Jae Sung Park, Jack Hessel, Khyathi Chandu, Paul Pu Liang, Ximing Lu, Peter West, Youngjae Yu, Qiuyuan Huang, Jianfeng Gao, Ali Farhadi, Yejin Choi

### Locally Invariant Explanations: Towards Stable and Unidirectional Explanations through Local Invariant Learning

**Authors:** Amit Dhurandhar, Karthikeyan Natesan Ramamurthy, Kartik Ahuja, Vijay Arya

### Lockdown: Backdoor Defense for Federated Learning  with Isolated Subspace Training

**Authors:** Tiansheng Huang, Sihao Hu, Ka-Ho Chow, Fatih Ilhan, Selim Tekin, Ling Liu

### Look Ma, No Hands! Agent-Environment Factorization of Egocentric Videos

**Authors:** Matthew Chang, Aditya Prakash, Saurabh Gupta

### Loss Dynamics of Temporal Difference Reinforcement Learning

**Authors:** Blake Bordelon, Paul Masset, Henry Kuo, Cengiz Pehlevan

### Low Tensor Rank Learning of Neural Dynamics

**Authors:** Arthur Pellegrino, N Alex Cayco Gajic, Angus Chadwick

### MIM4DD: Mutual Information Maximization for Dataset Distillation

**Authors:** Yuzhang Shang, Zhihang Yuan, Yan Yan

### Making Scalable Meta Learning Practical

**Authors:** Sang Choe, Sanket Vaibhav Mehta, Hwijeen Ahn, Willie Neiswanger, Pengtao Xie, Emma Strubell, Eric Xing

### Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack

**Authors:** Pratik Karmakar, Debabrota Basu

### MarioGPT: Open-Ended Text2Level Generation through Large Language Models

**Authors:** Shyam Sudhakaran, Miguel González-Duque, Matthias Freiberger, Claire Glanois, Elias Najarro, Sebastian Risi

### Mass-Producing Failures of Multimodal Systems with Language Models

**Authors:** Shengbang Tong, Erik Jones, Jacob Steinhardt

### MathNAS: If Blocks Have a Role in Mathematical Architecture Design

**Authors:** Qinsi Wang, Jinghan Ke, Zhi Liang, Sihai Zhang

### [Spotlight] Max-Margin Token Selection in Attention Mechanism

**Authors:** Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, Samet Oymak

### Max-Sliced Mutual Information

**Authors:** Dor Tsur, Ziv Goldfeld, Kristjan Greenewald

### Maximum Average Randomly Sampled: A Scale Free and Non-parametric Algorithm for Stochastic Bandits

**Authors:** Masoud Moravej Khorasani, Erik Weyer

### [Oral] Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean

**Authors:** Spyridon Kondylatos, Ioannis Prapas, Gustau Camps-Valls, Ioannis Papoutsis

**Oral Presentation:** We, Dec 13, 13:30 -- Oral 4B

### Meta-AdaM: An Meta-Learned Adaptive Optimizer with Momentum for Few-Shot Learning

**Authors:** Siyuan Sun, Hongyang Gao

### Meta-Adapter: An Online Few-shot Learner for Vision-Language Model

**Authors:** cheng cheng, Lin Song, Ruoyi Xue, Hang Wang, Hongbin Sun, Yixiao Ge, Ying Shan

### Meta-Learning with Neural Bandit Scheduler

**Authors:** Yunzhe Qi, Yikun Ban, Tianxin Wei, Jiaru Zou, Huaxiu Yao, Jingrui He

### Minimax Risks and Optimal Procedures for Estimation under Functional Local Differential Privacy

**Authors:** Bonwoo Lee, Jeongyoun Ahn, Cheolwoo Park

### Mirror Diffusion Models for Constrained and Watermarked Generation

**Authors:** Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, Molei Tao

### Mixed-Initiative Multiagent Apprenticeship Learning for Human Training of Robot Teams

**Authors:** Esmaeil Seraj, Jerry Xiong, Mariah Schrum, Matthew Gombolay

### Mixture Weight Estimation and Model Prediction in Multi-source Multi-target Domain Adaptation

**Authors:** Yuyang Deng, Ilja Kuzborskij, Mehrdad Mahdavi

### MoVie: Visual Model-Based Policy Adaptation for View Generalization

**Authors:** Sizhe Yang, Yanjie Ze, Huazhe Xu

### Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder

**Authors:** Huiwon Jang, Jihoon Tack, Daewon Choi, Jongheon Jeong, Jinwoo Shin

### Model Shapley: Equitable Model Valuation with Black-box Access

**Authors:** Xinyi Xu, Thanh Lam, Chuan Sheng Foo, Bryan Kian Hsiang Low

### Model-Based Reparameterization Policy Gradient Methods: Theory and Practical Algorithms

**Authors:** Shenao Zhang, Boyi Liu, Zhaoran Wang, Tuo Zhao

### Model-enhanced Vector Index

**Authors:** Hailin Zhang, Yujing Wang, Qi Chen, Ruiheng Chang, Ting Zhang, Ziming Miao, Yingyan Hou, Yang Ding, Xupeng Miao, Haonan Wang, Bochen Pang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Qi Zhang, Fan Yang, Xing Xie, Mao Yang, Bin CUI

### MomentDiff: Generative Video Moment Retrieval from Random to Real

**Authors:** Pandeng Li, Chen-Wei Xie, Hongtao Xie, Liming Zhao, Lei Zhang, Yun Zheng, Deli Zhao, Yongdong Zhang

### MotionGPT: Human Motion as a Foreign Language

**Authors:** Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen

### Multi-Agent First Order Constrained Optimization in Policy Space

**Authors:** Youpeng Zhao, Yaodong Yang, Zhenbo Lu, Wengang Zhou, Houqiang Li

### Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation

**Authors:** Haoran Chen, Xintong Han, Zuxuan Wu, Yu-Gang Jiang

### Multi-task Graph Neural Architecture Search with Task-aware Collaboration and Curriculum

**Authors:** Yijian Qin, Xin Wang, Ziwei Zhang, Hong Chen, Wenwu Zhu

### Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions

**Authors:** Kai Tan, Pierre C Bellec

### Multiplication-Free Transformer Training via Piecewise Affine Operations

**Authors:** Atli Kosson, Martin Jaggi

### NAP: Neural 3D Articulated Object Prior

**Authors:** Jiahui Lei, Congyue Deng, William B Shen, Leonidas Guibas, Kostas Daniilidis

### NCDL:  A Framework for Deep Learning on non-Cartesian Lattices

**Authors:** Joshua Horacsek, Usman Alim

### NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks

**Authors:** Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon

### Natural Language Instruction-following with Task-related Language Development and Translation

**Authors:** Jing-Cheng Pang, Xin-Yu Yang, Si-Hang Yang, Xiong-Hui Chen, Yang Yu

### Near-optimal learning with average Hölder smoothness

**Authors:** Guy Kornowski, Steve Hanneke, Aryeh Kontorovich

### Nearly Optimal Bounds for Cyclic Forgetting

**Authors:** William Swartworth, Deanna Needell, Rachel Ward, Mark Kong, Halyun Jeong

### Nearly Optimal VC-Dimension and Pseudo-Dimension Bounds for Deep Neural Network Derivatives

**Authors:** Yahong Yang, Haizhao Yang, Yang Xiang

### Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming

**Authors:** Jacobus van der Linden, Mathijs de Weerdt, Emir Demirović

### Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization

**Authors:** Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, Zhenkun Wang

### [Spotlight] Neural Injective Functions for Multisets, Measures and Graphs via a Finite Witness Theorem

**Authors:** Tal Amir, Steven Gortler, Ilai Avni, Ravina Ravina, Nadav Dym

### Neural Lad: A Neural Latent Dynamics Framework for Times Series Modeling

**Authors:** ting li, Jianguo Li, Zhanxing Zhu

### Neural Sampling in Hierarchical Exponential-family Energy-based Models

**Authors:** Xingsi Dong, Si Wu

### NeuroGF: A Neural Representation for Fast Geodesic Distance and Path Queries

**Authors:** Qijian Zhang, Junhui Hou, Yohanes Adikusuma, Wenping Wang, Ying He

### No-regret Algorithms for Fair Resource Allocation

**Authors:** Abhishek Sinha, Ativ Joshi, Rajarshi Bhattacharjee, Cameron Musco, Mohammad Hajiesmaili

### Non-Rigid Shape Registration via Deep Functional Maps Prior

**Authors:** puhua jiang, Mingze Sun, Ruqi Huang

### Nonparametric Boundary Geometry in Physics Informed Deep Learning

**Authors:** Scott Cameron, Arnu Pretorius, S Roberts

### Nonparametric Identifiability of Causal Representations from Unknown Interventions

**Authors:** Julius von Kügelgen, Michel Besserve, Liang Wendong, Luigi Gresele, Armin Kekić, Elias Bareinboim, David Blei, Bernhard Schölkopf

### Nonparametric Teaching for Multiple Learners

**Authors:** Chen Zhang, Xiaofeng Cao, Weiyang Liu, Ivor Tsang, James Kwok

### [Spotlight] Normalizing flow neural networks by JKO scheme

**Authors:** Chen Xu, Xiuyuan Cheng, Yao Xie

### Offline Reinforcement Learning for Mixture-of-Expert Dialogue Management

**Authors:** Dhawal Gupta, Yinlam Chow, Azamat Tulepbergenov, Mohammad Ghavamzadeh, Craig Boutilier

### On Calibrating Diffusion Probabilistic Models

**Authors:** Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng

### On Generalization Bounds for Projective Clustering

**Authors:** Maria Sofia Bucarelli, Matilde Larsen, Chris Schwiegelshohn, Mads Toftrup

### On Imitation in Mean-field Games

**Authors:** Giorgia Ramponi, Pavel Kolev, Olivier Pietquin, Niao He, Mathieu Lauriere, Matthieu Geist

### On Learning Latent Models with Multi-Instance Weak Supervision

**Authors:** Kaifu Wang, Efthymia Tsamoura, Dan Roth

### On Measuring Fairness in Generative Models

**Authors:** Christopher Teo, Milad Abdollahzadeh, Ngai-Man (Man) Cheung

### On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes

**Authors:** Rajat Modi, Vibhav Vineet, Yogesh Rawat

### On Separate Normalization in Self-supervised Transformers

**Authors:** Xiaohui Chen, Yinkai Wang, Yuanqi Du, Soha Hassoun, Liping Liu

### On kernel-based statistical learning theory in the mean field limit

**Authors:** Christian Fiedler, Michael Herty, Sebastian Trimpe

### On the Ability of Graph Neural Networks to Model Interactions Between Vertices

**Authors:** Noam Razin, Tom Verbin, Nadav Cohen

### On the Convergence of Black-Box Variational Inference

**Authors:** Kyurae Kim, Jisu Oh, Kaiwen Wu, Yian Ma, Jacob Gardner

### On the Convergence of CART under Sufficient Impurity Decrease Condition

**Authors:** Rahul Mazumder, Haoyue Wang

### [Spotlight] On the Gini-impurity Preservation For Privacy Random Forests

**Authors:** XinRan Xie, Man-Jie Yuan, Xuetong Bai, Wei Gao, Zhi-Hua Zhou

### [Spotlight] On the Learnability of Multilabel Ranking

**Authors:** Vinod Raman, UNIQUE SUBEDI, Ambuj Tewari

### On the Robustness of Removal-Based Feature Attributions

**Authors:** Chris Lin, Ian Covert, Su-In Lee

### On the Role of Entanglement and Statistics in Learning

**Authors:** Srinivasan Arunachalam, Vojtech Havlicek, Louis Schatzki

### [Spotlight] On the Role of Randomization in Adversarially Robust Classification

**Authors:** Lucas Gnecco Heredia, Muni Sreenivas Pydi, Laurent Meunier, Benjamin Negrevergne, Yann Chevaleyre

### On the Size and Approximation Error of Distilled Datasets

**Authors:** Alaa Maalouf, Murad Tukan, Noel Loo, Ramin Hasani, Mathias Lechner, Daniela Rus

### On-the-Fly Adapting Code Summarization on Trainable Cost-Effective Language Models

**Authors:** Yufan Cai, Yun Lin, Chenyan Liu, Jinglian Wu, Yifan Zhang, Yiming Liu, Yeyun Gong, Jin Song Dong

### One-Step Diffusion Distillation via Deep Equilibrium Models

**Authors:** Zhengyang Geng, Ashwini Pokle, J. Zico Kolter

### [Spotlight] Online Control for Meta-optimization

**Authors:** Xinyi Chen, Elad Hazan

### Online Inventory Problems: Beyond the i.i.d. Setting with Online Convex Optimization

**Authors:** Massil HIHAT, Stéphane Gaïffas, Guillaume Garrigos, Simon Bussy

### Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection

**Authors:** Chao Chen, Zhihang Fu, Kai Liu, Ze Chen, Mingyuan Tao, Jieping Ye

### Optimal Unbiased Randomizers for Regression with Label Differential Privacy

**Authors:** Badih Ghazi, Pritish Kamath, Ravi Kumar, Ethan Leeman, Pasin Manurangsi, Avinash V Varadarajan, Chiyuan Zhang

### Optimal and Fair Encouragement Policy Evaluation and Learning

**Authors:** Angela Zhou

### Optimistic Meta-Gradients

**Authors:** Sebastian Flennerhag, Tom Zahavy, Brendan O'Donoghue, Hado van Hasselt, András György, Satinder Singh

### PAC-Bayes Generalization Certificates for Learned Inductive Conformal Prediction

**Authors:** Apoorva Sharma, Sushant Veer, Asher Hancock, Heng Yang, Marco Pavone, Anirudha Majumdar

### PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning

**Authors:** Hojoon Lee, Hanseul Cho, HYUNSEUNG KIM, DAEHOON GWAK, Joonkee Kim, Jaegul Choo, Se-Young Yun, Chulhee Yun

### PROTES: Probabilistic Optimization with Tensor Sampling

**Authors:** Anastasiia Batsheva, Andrei Chertkov, Gleb Ryzhakov, Ivan Oseledets

### PUCA: Patch-Unshuffle and Channel Attention for Enhanced Self-Supervised Image Denoising

**Authors:** Hyemi Jang, Junsung Park, Dahuin Jung, Jaihyun Lew, Ho Bae, Sungroh Yoon

### PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas

**Authors:** Zheng Chen, Yan-Pei Cao, Yuan-Chen Guo, Chen Wang, Ying Shan, Song-Hai Zhang

### PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation

**Authors:** Jialu Li, Mohit Bansal

### [Spotlight] Parallel Sampling of Diffusion Models

**Authors:** Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari

### Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models

**Authors:** Qiong Wu, Wei Yu, Yiyi Zhou, Shubin Huang, Xiaoshuai Sun, Rongrong Ji

### Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

**Authors:** Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer

### [Spotlight] Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and Luck

**Authors:** Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, Cyril Zhang

### Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

**Authors:** Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey Gritsenko, Mario Lucic, Neil Houlsby

### Path Regularization: A Convexity and Sparsity Inducing Regularization for Parallel ReLU Networks

**Authors:** Tolga Ergen, Mert Pilanci

### [Spotlight] Paxion: Patching Action Knowledge in Video-Language Foundation Models

**Authors:** Zhenhailong Wang, Ansel Blume, Sha Li, Genglin Liu, Jaemin Cho, Zineng Tang, Mohit Bansal, Heng Ji

### Perceptual Kalman Filters: Online State Estimation under a Perfect Perceptual-Quality Constraint

**Authors:** Dror Freirich, Tomer Michaeli, Ron Meir

### Performance-optimized deep neural networks are evolving into worse models of inferotemporal visual cortex

**Authors:** Drew Linsley, Ivan F Rodriguez Rodriguez, Thomas FEL, Michael Arcaro, Saloni Sharma, Margaret Livingstone, Thomas Serre

### PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models

**Authors:** Jiacheng Chen, Ruizhi Deng, Yasutaka Furukawa

### Polynomial-Time Linear-Swap Regret Minimization in Imperfect-Information Sequential Games

**Authors:** Gabriele Farina, Charilaos Pipis

### [Spotlight] Posterior Contraction Rates for Matérn Gaussian Processes on Riemannian Manifolds

**Authors:** Paul Rosa, Slava Borovitskiy, Alexander Terenin, Judith Rousseau

### Posterior Sampling for Competitive RL: Function Approximation and Partial Observation

**Authors:** Shuang Qiu, Ziyu Dai, Han Zhong, Zhaoran Wang, Zhuoran Yang, Tong Zhang

### Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation

**Authors:** Nikki Lijing Kuang, Ming Yin, Mengdi Wang, Yu-Xiang Wang, Yian Ma

### Predict-then-Calibrate: A New Perspective of Robust Contextual LP

**Authors:** Chunlin Sun, Linyu Liu, Xiaocheng Li

### Predicting a Protein's Stability under a Million Mutations

**Authors:** Jeffrey Ouyang-Zhang, Daniel Diaz, Adam Klivans, Philipp Kraehenbuehl

### [Spotlight] Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

**Authors:** Xiaoxiao Sun, Nidham Gazagnadou, Vivek Sharma, Lingjuan Lyu, Hongdong Li, Liang Zheng

### [Spotlight] Private (Stochastic) Non-Convex Optimization Revisited: Second-Order Stationary Points and Excess Risks

**Authors:** Daogao Liu, Arun Ganesh, Sewoong Oh, Abhradeep Guha Thakurta

### Probabilistic Invariant Learning with Randomized Linear Classifiers

**Authors:** Leonardo Cotta, Gal Yehuda, Assaf Schuster, Chris Maddison

### Projection-Free Methods for Solving Nonconvex-Concave Saddle Point Problems

**Authors:** Morteza Boroun, Erfan Yazdandoost Hamedani, Afrooz Jalilzadeh

### [Spotlight] Promises and Pitfalls of Threshold-based Auto-labeling

**Authors:** Harit Vishwakarma, Heguang Lin, Frederic Sala, Ramya Korlakai Vinayak

### [Spotlight] Provable Training for Graph Contrastive Learning

**Authors:** Yue Yu, Xiao Wang, Mengmei Zhang, Nian Liu, Chuan Shi

### Public Opinion Field Effect Fusion in Representation Learning for Trending Topics Diffusion

**Authors:** Junliang Li, Yang Yajun, Qinghua Hu, Xin Wang, Hong Gao

### [Spotlight] Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving

**Authors:** Sepidehsadat (Sepid) Hossieni, Mohammad Amin Shabani, Saghar Irandoust, Yasutaka Furukawa

### PyNeRF: Pyramidal Neural Radiance Fields

**Authors:** Haithem Turki, Michael Zollhöfer, Christian Richardt, Deva Ramanan

### Quantum Bayesian Optimization

**Authors:** Zhongxiang Dai, Gregory Kang Ruey Lau, Arun Verma, YAO SHU, Bryan Kian Hsiang Low, Patrick Jaillet

### [Oral] Quilt-1M: One Million Image-Text Pairs for Histopathology

**Authors:** Wisdom Ikezogwo, Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, Linda Shapiro

**Oral Presentation:** We, Dec 13, 14:00 -- Oral 4B

### RDumb: A simple approach that questions our progress in continual test-time adaptation

**Authors:** Ori Press, Steffen Schneider, Matthias Kümmerer, Matthias Bethge

### RECKONING: Reasoning through Dynamic Knowledge Encoding

**Authors:** Zeming Chen, Gail Weiss, Eric Mitchell, Asli Celikyilmaz, Antoine Bosselut

### REx: Data-Free Residual Quantization Error Expansion

**Authors:** Edouard YVINEC, Arnaud Dapogny, Matthieu Cord, Kevin Bailly

### RanPAC: Random Projections and Pre-trained Models for Continual Learning

**Authors:** Mark D. McDonnell, Dong Gong, Amin Parvaneh, Ehsan Abbasnejad, Anton van den Hengel

### Rank-DETR for High Quality Object Detection

**Authors:** Yifan Pu, Weicong Liang, Yiduo Hao, YUHUI YUAN, Yukang Yang, Chao Zhang, Han Hu, Gao Huang

### [Spotlight] Rank-N-Contrast: Learning Continuous Representations for Regression

**Authors:** Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, Dina Katabi

### RayDF: Neural Ray-surface Distance Fields with Multi-view Consistency

**Authors:** Zhuoman Liu, Bo Yang, Yan Luximon, Ajay Kumar, Jinxi Li

### ReMaX: Relaxing for Better Training on Efficient Panoptic Segmentation

**Authors:** Shuyang Sun, WEIJUN WANG, Andrew Howard, Qihang Yu, Philip Torr, Liang-Chieh Chen

### ReSync: Riemannian Subgradient-based Robust Rotation Synchronization

**Authors:** Huikang Liu, Xiao Li, Anthony Man-Cho So

### Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding

**Authors:** Zhejun Zhang, Alexander Liniger, Christos Sakaridis, Fisher Yu, Luc V Gool

### Regret Minimization via Saddle Point Optimization

**Authors:** Johannes Kirschner, Alireza Bakhtiari, Kushagra Chandak, Volodymyr Tkachuk, Csaba Szepesvari

### [Spotlight] Regularized Behavior Cloning for Blocking the Leakage of Past Action Information

**Authors:** Seokin Seo, HyeongJoo Hwang, Hongseok Yang, Kee-Eung Kim

### Regularizing Neural Networks with Meta-Learning Generative Models

**Authors:** Shin'ya Yamaguchi, Daiki Chijiwa, Sekitoshi Kanai, Atsutoshi Kumagai, Hisashi Kashima

### Rehearsal Learning for Avoiding Undesired Future

**Authors:** Tian Qin, Tian-Zuo Wang, Zhi-Hua Zhou

### [Spotlight] Relax, it doesn’t matter how you get there: A new self-supervised approach for multi-timescale behavior analysis

**Authors:** Mehdi Azabou, Michael Mendelson, Nauman Ahad, Maks Sorokin, Shantanu Thakoor, Carolina Urzay, Eva Dyer

### Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective

**Authors:** Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, Yixuan Su

### Reproducibility in Multiple Instance Learning: A Case For Algorithmic Unit Tests

**Authors:** Edward Raff, James Holt

### Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone

**Authors:** Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Ao Ma, Yiliang Lv, Yujun Shen, Deli Zhao, Jingren Zhou

### Reusing Pretrained Models by Multi-linear Operators for Efficient Training

**Authors:** Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, Qun Liu

### RevColV2: Exploring Disentangled Representations in Masked Image Modeling

**Authors:** Qi Han, Yuxuan Cai, Xiangyu Zhang

### Revisiting Area Convexity: Faster Box-Simplex Games and Spectrahedral Generalizations

**Authors:** Arun Jambulapati, Kevin Tian

### Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification

**Authors:** Tianjun Ke, Haoqun Cao, Zenan Ling, Feng Zhou

### Riemannian Laplace approximations for Bayesian neural networks

**Authors:** Federico Bergamin, Pablo Moreno-Muñoz, Søren Hauberg, Georgios Arvanitidis

### Riemannian Residual Neural Networks

**Authors:** Isay Katsman, Eric Chen, Sidhanth Holalkere, Anna Asch, Aaron Lou, Ser Nam Lim, Christopher De Sa

### Risk-Averse Active Sensing for Timely Outcome Prediction under Cost Pressure

**Authors:** Yuchao Qin, Mihaela van der Schaar, Changhee Lee

### Robust and Actively Secure Serverless Collaborative Learning

**Authors:** Nicholas Franzese, Adam Dziedzic, Christopher A. Choquette-Choo, Mark R Thomas, Muhammad Ahmad Kaleem, Stephan Rabanser, Congyu Fang, Somesh Jha, Nicolas Papernot, Xiao Wang

### SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation

**Authors:** Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, Jian Yang

### SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models

**Authors:** Martin Gonzalez, Nelson Fernandez Pinto, Thuy Tran, elies Gherbi, Hatem Hajri, Nader Masmoudi

### SQ Lower Bounds for Non-Gaussian Component Analysis with Weaker Assumptions

**Authors:** Ilias Diakonikolas, Daniel Kane, Lisheng Ren, Yuxin Sun

### [Spotlight] STEVE-1: A Generative Model for Text-to-Behavior in Minecraft

**Authors:** Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, Sheila McIlraith

### STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning

**Authors:** Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, Gao Huang

### Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms

**Authors:** Akifumi Wachi, Wataru Hashimoto, Xun Shen, Kazumune Hashimoto

### [Spotlight] Safety Verification of Decision-Tree Policies in Continuous Time

**Authors:** Christian Schilling, Anna Lukina, Emir Demirović, Kim Larsen

### [Spotlight] Sample Complexity of Forecast Aggregation

**Authors:** Tao Lin, Yiling Chen

### Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents

**Authors:** Woojun Kim, Yongjae Shin, Jongeui Park, Youngchul Sung

### SatLM: Satisfiability-Aided Language Models Using Declarative Prompting

**Authors:** Xi Ye, Qiaochu Chen, Isil Dillig, Greg Durrett

### Scale-Space Hypernetworks for Efficient Biomedical Image Analysis

**Authors:** Jose Javier Gonzalez Ortiz, John Guttag, Adrian Dalca

### Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation

**Authors:** Wenhao Ding, Laixi Shi, Yuejie Chi, DING ZHAO

### Segment Anything in High Quality

**Authors:** Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu

### Segment Everything Everywhere All at Once

**Authors:** Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, Yong Jae Lee

### Self-Chained Image-Language Model for Video Localization and Question Answering

**Authors:** Shoubin Yu, Jaemin Cho, Prateek Yadav, Mohit Bansal

### Self-Refine: Iterative Refinement with Self-Feedback

**Authors:** Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark

### Self-Supervised Learning with Lie Symmetries for Partial Differential Equations

**Authors:** Grégoire Mialon, Quentin Garrido, Hannah Lawrence, Danyal Rehman, Yann LeCun, Bobak Kiani

### Self-Supervised Visual Acoustic Matching

**Authors:** Arjun Somayazulu, Changan Chen, Kristen Grauman

### Self-supervised Graph Neural Networks via Low-Rank Decomposition

**Authors:** Liang Yang, Runjie Shi, Qiuliang Zhang, bingxin niu, Zhen Wang, Xiaochun Cao, Chuan Wang

### [Spotlight] Separable Physics-Informed Neural Networks

**Authors:** Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, Eunbyung Park

### Sequential Predictive Two-Sample and Independence Testing

**Authors:** Aleksandr Podkopaev, Aaditya Ramdas

### Sequential Subset Matching for Dataset Distillation

**Authors:** JIAWEI DU, Qin Shi, Joey Tianyi Zhou

### Sharp Recovery Thresholds of Tensor PCA Spectral Algorithms

**Authors:** Michael Feldman, David Donoho

### [Spotlight] SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning

**Authors:** Yifan Yang, Peiyao Xiao, Kaiyi Ji

### SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization

**Authors:** Hao Dong, Ismail Nejjar, Han Sun, Eleni Chatzi, Olga Fink

### Simple and Asymmetric Graph Contrastive Learning without Augmentations

**Authors:** Teng Xiao, Huaisheng Zhu, Zhengyu Chen, Suhang Wang

### Simplifying and Empowering Transformers for Large-Graph Representations

**Authors:** Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, Junchi Yan

### SmooSeg: Smoothness Prior for Unsupervised Semantic Segmentation

**Authors:** Mengcheng Lan, Xinjiang Wang, Yiping Ke, Jiaxing Xu, Litong Feng, Wayne Zhang

### [Spotlight] Smoothed Online Learning for Prediction in Piecewise Affine Systems

**Authors:** Adam Block, Max Simchowitz, Russ Tedrake

### [Oral] Smoothing the Landscape Boosts the Signal for SGD: Optimal Sample Complexity for Learning Single Index Models

**Authors:** Alex Damian, Eshaan Nichani, Rong Ge, Jason Lee

**Oral Presentation:** We, Dec 13, 13:45 -- Oral 4A

### Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models

**Authors:** Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, Sanjay Shakkottai

### Solving a Class of Non-Convex  Minimax Optimization in Federated Learning

**Authors:** Xidong Wu, Jianhui Sun, Zhengmian Hu, Aidong Zhang, Heng Huang

### Sorting with Predictions

**Authors:** Xingjian Bai, Christian Coester

### Sparse Modular Activation for Efficient Sequence Modeling

**Authors:** Liliang Ren, Yang Liu, Shuohang Wang, Yichong Xu, Chenguang Zhu, Cheng Xiang Zhai

### Sparse Parameterization for Epitomic Dataset Distillation

**Authors:** Xing Wei, Anjia Cao, Funing Yang, Zhiheng Ma

### Spatio-Angular Convolutions for Super-resolution in Diffusion MRI

**Authors:** Matthew Lyon, Paul Armitage, Mauricio A Álvarez

### SpecTr: Fast Speculative Decoding via Optimal Transport

**Authors:** Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, Felix Yu

### Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning

**Authors:** Stefan Stojanovic, Yassir Jedra, Alexandre Proutiere

### Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts

**Authors:** Zeyang Zhang, Xin Wang, Ziwei Zhang, Zhou Qin, Weigao Wen, Hui Xue', Haoyang Li, Wenwu Zhu

### Stochastic Approximation Algorithms for Systems of Interacting Particles

**Authors:** Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause

### Strategic Apple Tasting

**Authors:** Keegan Harris, Chara Podimata, Steven Wu

### Strategic Behavior in Two-sided Matching Markets with Prediction-enhanced Preference-formation

**Authors:** Stefania Ionescu, Yuhao Du, Kenneth Joseph, Ancsa Hannak

### Structural Pruning for Diffusion Models

**Authors:** Gongfan Fang, Xinyin Ma, Xinchao Wang

### [Spotlight] Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data

**Authors:** Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, Shirui Pan

### Structured Neural Networks for Density Estimation and Causal Inference

**Authors:** Asic Chen, Ruian (Ian) Shi, Xiang Gao, Ricardo Baptista, Rahul Krishnan

### Structured Prediction with Stronger Consistency Guarantees

**Authors:** Anqi Mao, Mehryar Mohri, Yutao Zhong

### Structured Semidefinite Programming for Recovering Structured Preconditioners

**Authors:** Arun Jambulapati, Jerry Li, Christopher Musco, Kirankumar Shiragur, Aaron Sidford, Kevin Tian

### [Spotlight] SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks

**Authors:** Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj (Raj) Ammanabrolu, Yejin Choi, Xiang Ren

### Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning

**Authors:** Xiaoqian Wu, Yong-Lu Li, Jianhua Sun, Cewu Lu

### SyncTREE: Fast Timing Analysis for Integrated Circuit Design through a Physics-informed Tree-based Graph Neural Network

**Authors:** Yuting Hu, Jiajie Li, Florian Klemme, Gi-Joon Nam, Tengfei Ma, Hussam Amrouch, Jinjun Xiong

### TART: A plug-and-play Transformer module for task-agnostic reasoning

**Authors:** Kush Bhatia, Avanika Narayan, Christopher De Sa, Christopher Ré

### TD Convergence: An Optimization Perspective

**Authors:** Kavosh Asadi, Shoham Sabach, Yao Liu, Omer Gottesman, Rasool Fakoor

### Temporal Conditioning Spiking Latent Variable Models of the Neural Response to Natural Visual Scenes

**Authors:** Gehua Ma, Runhao Jiang, Rui Yan, Huajin Tang

### Temporally Disentangled Representation Learning under Unknown Nonstationarity

**Authors:** Xiangchen Song, Weiran Yao, Yewen Fan, Xinshuai Dong, Guangyi Chen, Juan Carlos Niebles, Eric Xing, Kun Zhang

### Test-Time Distribution Normalization for Contrastively Learned Visual-language Models

**Authors:** Yifei Zhou, Juntao Ren, Fengyu Li, Ramin Zabih, Ser Nam Lim

### Text Promptable Surgical Instrument Segmentation with Vision-Language Models

**Authors:** Zijian Zhou, Oluwatosin Alabi, Meng Wei, Tom Vercauteren, Miaojing Shi

### Textually Pretrained Speech Language Models

**Authors:** Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis CONNEAU, Felix Kreuk, Jade Copet, Alexandre Defossez, Gabriel Synnaeve, Emmanuel Dupoux, Roy Schwartz, Yossi Adi

### The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model

**Authors:** Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, Yuejie Chi

### The Distortion of Binomial Voting Defies Expectation

**Authors:** Yannai A. Gonczarowski, Gregory Kehne, Ariel Procaccia, Ben Schiffer, Shirley Zhang

### The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter

**Authors:** AJAY JAISWAL, Shiwei Liu, Tianlong Chen, Zhangyang "Atlas" Wang

### The Memory-Perturbation Equation: Understanding Model's Sensitivity to Data

**Authors:** Peter Nickl, Lu Xu, Dharmesh Tailor, Thomas Möllenhoff, Mohammad Emtiyaz Khan

### The Rank-Reduced Kalman Filter: Approximate Dynamical-Low-Rank Filtering In High Dimensions

**Authors:** Jonathan Schmidt, Philipp Hennig, Jörg Nick, Filip Tronarp

### [Spotlight] The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance

**Authors:** Jon Donnelly, Srikar Katta, Cynthia Rudin, Edward Browne

### The Utility of “Even if” Semifactual Explanation to Optimise Positive Outcomes

**Authors:** Eoin Kenny, Weipeng Huang

### The emergence of clusters in self-attention dynamics

**Authors:** Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet

### The geometry of hidden representations of large transformer models

**Authors:** Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, Alberto Cazzaniga

### The s-value: evaluating stability with respect to distributional shifts

**Authors:** Suyash Gupta, Dominik Rothenhäusler

### This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations

**Authors:** Chiyu Ma, Brandon Zhao, Chaofan Chen, Cynthia Rudin

### [Spotlight] Thought Cloning: Learning to Think while Acting by Imitating Human Thinking

**Authors:** Shengran Hu, Jeff Clune

### Tight Bounds for Volumetric Spanners and Applications

**Authors:** Aditya Bhaskara, Sepideh Mahabadi, Ali Vakilian

### Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings

**Authors:** Giovanni De Felice, John Goulermas, Vladimir Gusev

### Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

**Authors:** Zekun Li, Shiyang Li, Xifeng Yan

### Tools for Verifying Neural Models' Training Data

**Authors:** Dami Choi, Yonadav Shavit, David Duvenaud

### Top-Ambiguity Samples Matter: Understanding Why Deep Ensemble Works in Selective Classification

**Authors:** Qiang Ding, Yixuan Cao, Ping Luo

### [Spotlight] Topological Parallax: A Geometric Specification for Deep Perception Models

**Authors:** Abraham Smith, Michael Catanzaro, Gabrielle Angeloro, Nirav Patel, Paul Bendich

### Toward Understanding Generative Data Augmentation

**Authors:** Chenyu Zheng, Guoqiang Wu, Chongxuan LI

### Towards Consistent Video Editing with Text-to-Image Diffusion Models

**Authors:** Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, Luoqi Liu

### Towards Distribution-Agnostic Generalized Category Discovery

**Authors:** Jianhong Bai, Zuozhu Liu, Hualiang Wang, Ruizhe Chen, Lianrui Mu, Xiaomeng Li, Joey Tianyi Zhou, YANG FENG, Jian Wu, Haoji Hu

### Towards Efficient and Accurate Winograd Convolution via Full Quantization

**Authors:** Tianqi Chen, Weixiang Xu, Weihan Chen, Peisong Wang, Jian Cheng

### Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly

**Authors:** Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

### Towards Higher Ranks via Adversarial Weight Pruning

**Authors:** Yuchuan Tian, Hanting Chen, Tianyu Guo, Chao Xu, Yunhe Wang

### [Oral] Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective

**Authors:** Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, Liwei Wang

**Oral Presentation:** We, Dec 13, 14:15 -- Oral 4C

### Towards Robust and Expressive Whole-body Human Pose and Shape Estimation

**Authors:** Hui En Pang, Zhongang Cai, Lei Yang, Qingyi Tao, Zhonghua Wu, Tianwei Zhang, Ziwei Liu

### Towards Self-Interpretable Graph-Level Anomaly Detection

**Authors:** Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, Shirui Pan

### Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning

**Authors:** Chang Lu, Chandan Reddy, Ping Wang, Yue Ning

### Towards Test-Time Refusals via Concept Negation

**Authors:** Peiran Dong, Song Guo, Junxiao Wang, Bingjie WANG, Jiewei Zhang, Ziming Liu

### Tracking Most Significant Shifts in Nonparametric Contextual Bandits

**Authors:** Joe Suk, Samory Kpotufe

### Train Faster, Perform Better: Modular Adaptive Training in Over-Parameterized Models

**Authors:** Yubin Shi, Yixuan Chen, Mingzhi Dong, Xiaochen Yang, Dongsheng Li, Yujiang Wang, Robert Dick, Qin Lv, Yingying Zhao, Fan Yang, Tun Lu, Ning Gu, Li Shang

### Train Hard, Fight Easy: Robust Meta Reinforcement Learning

**Authors:** Ido Greenberg, Shie Mannor, Gal Chechik, Eli Meirom

### Training Neural Networks is NP-Hard in Fixed Dimension

**Authors:** Vincent Froese, Christoph Hertrich

### [Oral] Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

**Authors:** Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei

**Oral Presentation:** We, Dec 13, 14:00 -- Oral 4C

### Transportability for Bandits with Data from Different Environments

**Authors:** Alexis Bellot, Alan Malek, Silvia Chiappa

### [Spotlight] Tree Variational Autoencoders

**Authors:** Laura Manduchi, Moritz Vandenhirtz, Alain Ryser, Julia Vogt

### [Oral] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, Karthik Narasimhan

**Oral Presentation:** We, Dec 13, 13:45 -- Oral 4C

### TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models

**Authors:** Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Bölöni, Qian Lou

### Trust Your $\nabla$: Gradient-based Intervention Targeting for Causal Discovery

**Authors:** Mateusz Olko, Michał Zając, Aleksandra Nowak, Nino Scherrer, Yashas Annadani, Stefan Bauer, Łukasz Kuciński, Piotr Miłoś

### Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation

**Authors:** Shutong Ding, Tianyu Cui, Jingya Wang, Ye Shi

### UNSSOR: Unsupervised Neural Speech Separation by Leveraging Over-determined Training Mixtures

**Authors:** Zhong-Qiu Wang, Shinji Watanabe

### UP-DP: Unsupervised Prompt Learning for Data Pre-Selection with Vision-Language Models

**Authors:** Xin Li, Sima Behpour, Thang Long Doan, Wenbin He, Liang Gou, Liu Ren

### UP-NeRF: Unconstrained Pose Prior-Free Neural Radiance Field

**Authors:** Injae Kim, Minhyuk Choi, Hyunwoo Kim

### Unbalanced Low-rank Optimal Transport Solvers

**Authors:** Meyer Scetbon, Michal Klein, Giovanni Palla, Marco Cuturi

### [Spotlight] Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts

**Authors:** Pritam Sarkar, Ahmad Beirami, Ali Etemad

### Understanding Neural Network Binarization with Forward and Backward Proximal Quantizers

**Authors:** Yiwei Lu, Yaoliang Yu, Xinlin Li, Vahid Partovi Nia

### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

**Authors:** Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, Youngjung Uh

### Uniform Convergence with Square-Root Lipschitz Loss

**Authors:** Lijia Zhou, Zhen Dai, Frederic Koehler, Nati Srebro

### Unlocking Deterministic Robustness Certification on ImageNet

**Authors:** Kai Hu, Andy Zou, Zifan Wang, Klas Leino, Matt Fredrikson

### Unsupervised Anomaly Detection with Rejection

**Authors:** Lorenzo Perini, Jesse Davis

### Unsupervised Graph Neural Architecture Search with Disentangled Self-Supervision

**Authors:** Zeyang Zhang, Xin Wang, Ziwei Zhang, Guangyao Shen, Shiqi Shen, Wenwu Zhu

### Unsupervised Image Denoising with Score Function

**Authors:** Yutong Xie, Mingze Yuan, Bin Dong, Quanzheng Li

### Unsupervised Video Domain Adaptation for Action Recognition: A Disentanglement Perspective

**Authors:** Pengfei Wei, Lingdong Kong, Xinghua Qu, Yi Ren, Zhiqiang Xu, Jing Jiang, Xiang Yin

### VRA: Variational Rectified Activation for Out-of-distribution Detection

**Authors:** Mingyu Xu, Zheng Lian, Bin Liu, Jianhua Tao

### Variance-Reduced Gradient Estimation via Noise-Reuse in Online Evolution Strategies

**Authors:** Oscar Li, James Harrison, Jascha Sohl-Dickstein, Virginia Smith, Luke Metz

### Variational Annealing on Graphs for Combinatorial Optimization

**Authors:** Sebastian Sanokowski, Wilhelm Berghammer, Sepp Hochreiter, Sebastian Lehner

### VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks

**Authors:** Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, Jifeng Dai

### WCLD: Curated Large Dataset of Criminal Cases from Wisconsin Circuit Courts

**Authors:** Elliott Ash, Naman Goel, Nianyun Li, Claudia Marangon, Peiyao Sun

### Weighted ROC Curve in Cost Space: Extending AUC to Cost-Sensitive Learning

**Authors:** HuiYang Shao, Qianqian Xu, Zhiyong Yang, Peisong Wen, Gao Peifeng, Qingming Huang

### When can Regression-Adjusted Control Variate Help? Rare Events, Sobolev Embedding and Minimax Optimality

**Authors:** Jose Blanchet, Haoxuan Chen, Yiping Lu, Lexing Ying

### [Spotlight] Which Models have Perceptually-Aligned Gradients? An Explanation via Off-Manifold Robustness

**Authors:** Suraj Srinivas, Sebastian Bordt, Himabindu Lakkaraju

### White-Box Transformers via Sparse Rate Reduction

**Authors:** Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Benjamin Haeffele, Yi Ma

### [Oral] Why think step by step? Reasoning emerges from the locality of experience

**Authors:** Ben Prystawski, Michael Li, Noah Goodman

**Oral Presentation:** We, Dec 13, 13:30 -- Oral 4C

### Wide Neural Networks as Gaussian Processes: Lessons from Deep Equilibrium Models

**Authors:** Tianxiang Gao, Xiaokai Huo, Hailiang Liu, Hongyang Gao

### Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model

**Authors:** Zirui Liu, Guanchu Wang, Shaochen (Henry) Zhong, Zhaozhuo Xu, Daochen Zha, Ruixiang Tang, Zhimeng Jiang, Kaixiong Zhou, Vipin Chaudhary, Shuai Xu, Xia Hu

### Your representations are in the network: composable and parallel adaptation for large scale models

**Authors:** Yonatan Dukler, Alessandro Achille, Hao Yang, Varsha Vivek, Luca Zancato, Benjamin Bowman, Avinash Ravichandran, Charless Fowlkes, Ashwin Swaminathan, Stefano Soatto

### ZipLM: Inference-Aware Structured Pruning of Language Models

**Authors:** Eldar Kurtić, Elias Frantar, Dan Alistarh

</details>

<details><summary><h3 style='display: inline;'> Poster Session 5: Thursday, Dec 14, 08:45 CT</h3></summary>

### $H$-Consistency Bounds: Characterization and Extensions

**Authors:** Anqi Mao, Mehryar Mohri, Yutao Zhong

### $SE(3)$  Equivariant Convolution and Transformer in Ray Space

**Authors:** Yinshuang Xu, Jiahui Lei, Kostas Daniilidis

### $p$-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison

**Authors:** Kai Klede, Thomas Altstidl, Dario Zanca, Bjoern Eskofier

### (Provable) Adversarial Robustness for Group Equivariant Tasks: Graphs, Point Clouds, Molecules, and More

**Authors:** Jan Schuchardt, Yan Scholten, Stephan Günnemann

### 2Direction: Theoretically Faster Distributed Training with Bidirectional Communication Compression

**Authors:** Alexander Tyurin, Peter Richtarik

### A Causal Framework for Decomposing Spurious Variations

**Authors:** Drago Plecko, Elias Bareinboim

### A Computation and Communication Efficient Method for Distributed Nonconvex Problems in the Partial Participation Setting

**Authors:** Alexander Tyurin, Peter Richtarik

### A Finite-Sample Analysis of Payoff-Based Independent Learning in Zero-Sum Stochastic Games

**Authors:** Zaiwei Chen, Kaiqing Zhang, Eric Mazumdar, Asuman Ozdaglar, Adam Wierman

### A Fractional Graph Laplacian Approach to Oversmoothing

**Authors:** Sohir Maskey, Raffaele Paolino, Aras Bacho, Gitta Kutyniok

### A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions

**Authors:** David Loiseaux, Mathieu Carrière, Andrew Blumberg

### A General Framework for Robust G-Invariance in G-Equivariant Networks

**Authors:** Sophia Sanborn, Nina Miolane

### A Hierarchical Training Paradigm for Antibody Structure-sequence Co-design

**Authors:** Fang Wu, Stan Z. Li

### A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints

**Authors:** Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck

### [Oral] A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods

**Authors:** Veit David Wild, Sahra Ghalebikesabi, Dino Sejdinovic, Jeremias Knoblauch

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5C

### A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models

**Authors:** Alexander Reisach, Myriam Tami, Christof Seiler, Antoine Chambaz, Sebastian Weichwald

### A Single 2D Pose with Context is Worth Hundreds for 3D Human Pose Estimation

**Authors:** Qitao Zhao, Ce Zheng, Mengyuan Liu, Chen Chen

### A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence

**Authors:** Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, Ming-Hsuan Yang

### A Theoretical Analysis of the Test Error of Finite-Rank Kernel Ridge Regression

**Authors:** Tin Sum Cheng, Aurelien Lucchi, Anastasis Kratsios, Ivan Dokmanić, David Belius

### A Trichotomy for Transductive Online Learning

**Authors:** Steve Hanneke, Shay Moran, Jonathan Shafer

### A Unified Fast Gradient Clipping Framework for DP-SGD

**Authors:** Weiwei Kong, Andres Munoz Medina

### A Unified Model and Dimension for Interactive Estimation

**Authors:** Nataly Brukhim, Miro Dudik, Aldo Pacchiano, Robert Schapire

### A fast heuristic to optimize time-space tradeoff for large models

**Authors:** Akifumi Imanishi, Zijian Xu, Masayuki Takagi, Sixue Wang, Emilio Castillo

### A new perspective on building efficient and expressive 3D equivariant graph neural networks

**Authors:** weitao Du, Yuanqi Du, Limei Wang, Dieqiao Feng, Guifeng Wang, Shuiwang Ji, Carla Gomes, Zhi-Ming Ma

### A polar prediction model for learning to represent visual transformations

**Authors:** Pierre-Étienne Fiquet, Eero Simoncelli

### AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset

**Authors:** Jiakang Yuan, Bo Zhang, Xiangchao Yan, Botian Shi, Tao Chen, Yikang LI, Yu Qiao

### [Spotlight] AIMS: All-Inclusive Multi-Level Segmentation for Anything

**Authors:** Lu Qi, Jason Kuen, Weidong Guo, Jiuxiang Gu, Zhe Lin, Bo Du, Yu Xu, Ming-Hsuan Yang

### [Spotlight] AMDP: An Adaptive Detection Procedure for False Discovery Rate Control in High-Dimensional Mediation Analysis

**Authors:** Jiarong Ding, Xuehu ZHU

### ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections

**Authors:** Chun-Han Yao, Amit Raj, Wei-Chih Hung, Michael Rubinstein, Yuanzhen Li, Ming-Hsuan Yang, Varun Jampani

### AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web

**Authors:** Michael Schlichtkrull, Zhijiang Guo, Andreas Vlachos

### Active Observing in Continuous-time Control

**Authors:** Samuel Holt, Alihan Hüyük, Mihaela van der Schaar

### Active Vision Reinforcement Learning under Limited Visual Observability

**Authors:** Jinghuan Shang, Michael Ryoo

### Activity Grammars for Temporal Action Segmentation

**Authors:** Dayoung Gong, Joonseok Lee, Deunsol Jung, Suha Kwak, Minsu Cho

### AdaPlanner: Adaptive Planning from Feedback with Language Models

**Authors:** Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, Chao Zhang

### AdaVAE: Bayesian Structural Adaptation for Variational Autoencoders

**Authors:** Paribesh Regmi, Rui Li

### Adaptive Online Replanning with Diffusion Models

**Authors:** Siyuan Zhou, Yilun Du, Shun Zhang, Mengdi Xu, Yikang Shen, Wei Xiao, Dit-Yan Yeung, Chuang Gan

### Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels

**Authors:** Vijay Veerabadran, Srinivas Ravishankar, Yuan Tang, Ritik Raina, Virginia de Sa

### [Spotlight] Adversarial Counterfactual Environment Model Learning

**Authors:** Xiong-Hui Chen, Yang Yu, Zhengmao Zhu, ZhiHua Yu, Chen Zhenjun, Chenghe Wang, Yinan Wu, Rong-Jun Qin, Hongqiu Wu, Ruijin Ding, Huang Fangsheng

### Adversarial Examples Are Not Real Features

**Authors:** Ang Li, Yifei Wang, Yiwen Guo, Yisen Wang

### Adversarial Learning for Feature Shift Detection and Correction

**Authors:** Míriam Barrabés, Daniel Mas Montserrat, Margarita Geleta, Xavier Giró-i-Nieto, Alexander Ioannidis

### [Spotlight] Adversarial Training from Mean Field Perspective

**Authors:** Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

### Affinity-Aware Graph Networks

**Authors:** Ameya Velingker, Ali Sinop, Ira Ktena, Petar Veličković, Sreenivas Gollapudi

### Aggregating Capacity in FL through Successive Layer Training for Computationally-Constrained Devices

**Authors:** Kilian Pfeiffer, Ramin Khalili, Joerg Henkel

### Algorithmic Regularization in Tensor Optimization: Towards a Lifted Approach in Matrix Sensing

**Authors:** Ziye Ma, Javad Lavaei, Somayeh Sojoudi

### [Spotlight] Alleviating the Semantic Gap for Generalized fMRI-to-Image Reconstruction

**Authors:** Tao Fang, Qian Zheng, Gang Pan

### An Adaptive Algorithm for Learning with Unknown Distribution Drift

**Authors:** Alessio Mazzetto, Eli Upfal

### An active learning framework for multi-group mean estimation

**Authors:** Abdellah Aznag, Rachel Cummings, Adam N. Elmachtoub

### An information-theoretic quantification of the content of communication between brain regions

**Authors:** Marco Celotto, Jan Bím, Alejandro Tlaie, Vito De Feo, Alessandro Toso, Stefan Lemke, Daniel Chicharro, Hamed Nili, Malte Bieler, Ileana Hanganu-Opatz, Tobias Donner, Andrea Brovelli, Stefano Panzeri

### Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods

**Authors:** Tobit Klug, Dogukan Atik, Reinhard Heckel

### Are Diffusion Models Vision-And-Language Reasoners?

**Authors:** Benno Krojer, Elinor Poole-Dayan, Vikram Voleti, Chris Pal, Siva Reddy

### Asynchronous Proportional Response Dynamics: Convergence in Markets with Adversarial Scheduling

**Authors:** Yoav Kolumbus, Menahem Levy, Noam Nisan

### AutoGO: Automated Computation Graph Optimization for Neural Network Evolution

**Authors:** Mohammad Salameh, Keith Mills, Negar Hassanpour, Fred Han, Shuting Zhang, Wei Lu, Shangling Jui, CHUNHUA ZHOU, Fengyu Sun, Di Niu

### Auxiliary Losses for Learning Generalizable Concept-based Models

**Authors:** Ivaxi Sheth, Samira Ebrahimi Kahou

### BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning

**Authors:** Xuan Chen, Wenbo Guo, Guanhong Tao, Xiangyu Zhang, Dawn Song

### BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing

**Authors:** DONGXU LI, Junnan Li, Steven Hoi

### Bayesian Active Causal Discovery with Multi-Fidelity Experiments

**Authors:** Zeyu Zhang, Chaozhuo Li, Xu Chen, Xing Xie

### Bayesian nonparametric (non-)renewal processes for analyzing neural spike train variability

**Authors:** David Liu, Mate Lengyel

### [Spotlight] Best Arm Identification with Fixed Budget: A Large Deviation Perspective

**Authors:** Po-An Wang, Ruo-Chun Tzeng, Alexandre Proutiere

### Better with Less: A Data-Active Perspective on Pre-Training Graph Neural Networks

**Authors:** Jiarong Xu, Renhong Huang, XIN JIANG, Yuxuan Cao, Carl Yang, Chunping Wang, YANG YANG

### Beyond Average Return in Markov Decision Processes

**Authors:** Alexandre Marthe, Aurélien Garivier, Claire Vernade

### Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift

**Authors:** Florian Seligmann, Philipp Becker, Michael Volpp, Gerhard Neumann

### Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense

**Authors:** Zunzhi You, Daochang Liu, Bohyung Han, Chang Xu

### BiSLS/SPS: Auto-tune Step Sizes for Stable Bi-level Optimization

**Authors:** Chen Fan, Gaspard Choné-Ducasse, Mark Schmidt, Christos Thrampoulidis

### Bicriteria Approximation Algorithms for the Submodular Cover Problem

**Authors:** Wenjing Chen, Victoria Crawford

### Binarized Spectral Compressive Imaging

**Authors:** Yuanhao Cai, Yuxin Zheng, Jing Lin, Xin Yuan, Yulun Zhang, Haoqian Wang

### Black-Box Differential Privacy for Interactive ML

**Authors:** Haim Kaplan, Yishay Mansour, Shay Moran, Kobbi Nissim, Uri Stemmer

### Block Broyden's Methods for Solving Nonlinear Equations

**Authors:** Chengchang Liu, Cheng Chen, Luo Luo, John C.S. Lui

### Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences

**Authors:** Minsu Kim, Federico Berto, Sungsoo Ahn, Jinkyoo Park

### Brain Dissection: fMRI-trained Networks Reveal Spatial Selectivity in the Processing of Natural Images

**Authors:** Gabriel Sarch, Michael Tarr, Katerina Fragkiadaki, Leila Wehbe

### [Spotlight] Break It Down:  Evidence for Structural Compositionality in Neural Networks

**Authors:** Michael Lepori, Thomas Serre, Ellie Pavlick

### Bypassing the Simulator: Near-Optimal Adversarial Linear Contextual Bandits

**Authors:** Haolin Liu, Chen-Yu Wei, Julian Zimmert

### C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models

**Authors:** Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, jiayi lei, Yao Fu, Maosong Sun, Junxian He

### CARE: Modeling Interacting Dynamics Under Temporal Environmental Variation

**Authors:** Xiao Luo, Haixin Wang, Zijie Huang, Huiyu Jiang, Abhijeet Gangan, Song Jiang, Yizhou Sun

### CAST: Cross-Attention in Space and Time for Video Action Recognition

**Authors:** Dongho Lee, Jongseo Lee, Jinwoo Choi

### CLIP4HOI: Towards Adapting CLIP for Practical Zero-Shot HOI Detection

**Authors:** Yunyao Mao, Jiajun Deng, Wengang Zhou, Li Li, Yao Fang, Houqiang Li

### CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence

**Authors:** Bong Gyun Kang, HyunGi Kim, Dahuin Jung, Sungroh Yoon

### CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

**Authors:** Anthony Fuller, Koreen Millard, James Green

### CWCL: Cross-Modal Transfer with Continuously Weighted Contrastive Loss

**Authors:** Rakshith Sharma Srinivasa, Jaejin Cho, Chouchang Yang, Yashas Malur Saidutta, Ching-Hua Lee, Yilin Shen, Hongxia Jin

### [Spotlight] Calibrated Stackelberg Games: Learning Optimal Commitments Against Calibrated Agents

**Authors:** Nika Haghtalab, Chara Podimata, Kunhe Yang

### Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?

**Authors:** Jialu Gao, Kaizhe Hu, Guowei Xu, Huazhe Xu

### Causal Discovery from Subsampled Time Series with Proxy Variables

**Authors:** Mingzhou Liu, Xinwei Sun, Lingjing Hu, Yizhou Wang

### Causal Effect Identification in Uncertain Causal Networks

**Authors:** Sina Akbari, Fateme Jamshidi, Ehsan Mokhtarian, Matthew Vowels, Jalal Etesami, Negar Kiyavash

### Causal de Finetti: On the Identification of Invariant Causal Structure in Exchangeable Data

**Authors:** Siyuan Guo, Viktor Toth, Bernhard Schölkopf, Ferenc Huszar

### Certifiably Robust Graph Contrastive Learning

**Authors:** Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang

### ChimpACT: A Longitudinal Dataset for Understanding Chimpanzee Behaviors

**Authors:** Xiaoxuan Ma, Stephan Kaufhold, Jiajun Su, Wentao Zhu, Jack Terwilliger, Andres Meza, Yixin Zhu, Federico Rossano, Yizhou Wang

### Class-Conditional Conformal Prediction with Many Classes

**Authors:** Tiffany Ding, Anastasios Angelopoulos, Stephen Bates, Michael Jordan, Ryan Tibshirani

### Classification of Heavy-tailed Features in High Dimensions: a Superstatistical Approach

**Authors:** Urte Adomaityte, Gabriele Sicuro, Pierpaolo Vivo

### [Oral] Clifford Group Equivariant Neural Networks

**Authors:** David Ruhe, Johannes Brandstetter, Patrick Forré

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5A

### CluB: Cluster Meets BEV for LiDAR-Based 3D Object Detection

**Authors:** Yingjie Wang, Jiajun Deng, Yuenan Hou, Yao Li, Yu Zhang, Jianmin Ji, Wanli Ouyang, Yanyong Zhang

### CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference

**Authors:** Wenxuan Zeng, Meng Li, Haichuan Yang, Wen-jie Lu, Runsheng Wang, Ru Huang

### [Spotlight] Coherent Soft Imitation Learning

**Authors:** Joe Watson, Sandy Huang, Nicolas Heess

### Collaborative Alignment of NLP Models

**Authors:** Fereshte Khani, Marco Tulio Ribeiro

### Combating Bilateral Edge Noise for Robust Link Prediction

**Authors:** Zhanke Zhou, Jiangchao Yao, Jiaxu Liu, Xiawei Guo, Quanming Yao, LI He, Liang Wang, Bo Zheng, Bo Han

### Combating Representation Learning Disparity with Geometric Harmonization

**Authors:** Zhihan Zhou, Jiangchao Yao, Feng Hong, Ya Zhang, Bo Han, Yanfeng Wang

### Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints

**Authors:** Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, Yangqiu Song

### Complex-valued Neurons Can Learn More but Slower than Real-valued Neurons via Gradient Descent

**Authors:** Jin-Hui Wu, Shao-Qun Zhang, Yuan Jiang, Zhi-Hua Zhou

### Composable Coresets for Determinant Maximization: Greedy is Almost Optimal

**Authors:** Siddharth Gollapudi, Sepideh Mahabadi, Varun Sivashankar

### Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task

**Authors:** Maya Okawa, Ekdeep S Lubana, Robert Dick, Hidenori Tanaka

### Computing Approximate $\ell_p$ Sensitivities

**Authors:** Swati Padmanabhan, David Woodruff, Richard Zhang

### Computing Optimal Nash Equilibria in Multiplayer Games

**Authors:** Youzhi Zhang, Bo An, Venkatramanan Subrahmanian

### Content-based Unrestricted Adversarial Attack

**Authors:** Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

### Contextual Gaussian Process Bandits with Neural Networks

**Authors:** Haoting Zhang, Jinghai He, Rhonda Righter, Zuo-Jun Shen, Zeyu Zheng

### Continual Learning for Instruction Following from Realtime Feedback

**Authors:** Alane Suhr, Yoav Artzi

### Continuous-time Analysis of Anchor Acceleration

**Authors:** Jaewook Suh, Jisun Park, Ernest Ryu

### Contrast Everything: A Hierarchical Contrastive Framework for Medical Time-Series

**Authors:** Yihe Wang, Yu Han, Haishuai Wang, Xiang Zhang

### Contrastive Retrospection: honing in on critical steps for rapid learning and generalization in RL

**Authors:** Chen Sun, Wannan Yang, Thomas Jiralerspong, Dane Malenfant, Benjamin Alsbury-Nealy, Yoshua Bengio, Blake Richards

### Contrastive Training of Complex-Valued Autoencoders for Object Discovery

**Authors:** Aleksandar Stanić, Anand Gopalakrishnan, Kazuki Irie, Jürgen Schmidhuber

### Convolutional Visual Prompt for Robust Visual Perception

**Authors:** Yun-Yun Tsai, Chengzhi Mao, Junfeng Yang

### [Spotlight] Coop: Memory is not a Commodity

**Authors:** Jianhao Zhang, Shihan Ma, Peihong Liu, Jinhui Yuan

### CorresNeRF: Image Correspondence Priors for Neural Radiance Fields

**Authors:** Yixing Lao, Xiaogang Xu, zhipeng cai, Xihui Liu, Hengshuang Zhao

### [Spotlight] Counterfactual Evaluation of Peer-Review Assignment Policies

**Authors:** Martin Saveski, Steven Jecmen, Nihar Shah, Johan Ugander

### Creating Multi-Level Skill Hierarchies in Reinforcement Learning

**Authors:** Joshua B. Evans, Özgür Şimşek

### Creating a Public Repository for Joining Private Data

**Authors:** James Cook, Milind Shyani, Nina Mishra

### Cross-Domain Policy Adaptation via Value-Guided Data Filtering

**Authors:** Kang Xu, Chenjia Bai, Xiaoteng Ma, Dong Wang, Bin Zhao, Zhen Wang, Xuelong Li, Wei Li

### Cross-links Matter for Link Prediction: Rethinking the Debiased GNN from a Data Perspective

**Authors:** Zihan Luo, Hong Huang, Jianxun Lian, Xiran Song, Xing Xie, Hai Jin

### Curriculum Learning for Graph Neural Networks: Which Edges Should We Learn First

**Authors:** Zheng Zhang, Junxiang Wang, Liang Zhao

### D-CIPHER: Discovery of Closed-form Partial Differential Equations

**Authors:** Krzysztof Kacprzyk, Zhaozhi Qian, Mihaela van der Schaar

### DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models

**Authors:** Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, Sibei Yang

### DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries

**Authors:** Joshua Engels, Benjamin Coleman, Vihan Lakshman, Anshumali Shrivastava

### DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework

**Authors:** Siran Dai, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang

### DaTaSeg: Taming a Universal Multi-Dataset Multi-Task Segmentation Model

**Authors:** Xiuye Gu, Yin Cui, Jonathan Huang, Abdullah Rashwan, Xuan Yang, Xingyi Zhou, Golnaz Ghiasi, Weicheng Kuo, Huizhong Chen, Liang-Chieh Chen, David Ross

### [Oral] DataComp: In search of the next generation of multimodal datasets

**Authors:** Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5D

### Debiasing Scores and Prompts of 2D Diffusion for View-consistent Text-to-3D Generation

**Authors:** Susung Hong, Donghoon Ahn, Seungryong Kim

### Deconstructing Data Reconstruction: Multiclass, Weight Decay and General Losses

**Authors:** Gon Buzaglo, Niv Haim, Gilad Yehudai, Gal Vardi, Yakir Oz, Yaniv Nikankin, Michal Irani

### [Spotlight] Deep Reinforcement Learning with Plasticity Injection

**Authors:** Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, Andre Barreto

### Demo2Code: From Summarizing Demonstrations to Synthesizing Code via Extended Chain-of-Thought

**Authors:** Yuki Wang, Gonzalo Gonzalez-Pumariega, Yash Sharma, Sanjiban Choudhury

### Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?

**Authors:** Haitao Mao, Zhikai Chen, Wei Jin, Haoyu Han, Yao Ma, Tong Zhao, Neil Shah, Jiliang Tang

### Demystifying the Optimal Performance of Multi-Class Classification

**Authors:** Minoh Jeong, Martina Cardone, Alex Dytso

### Design from Policies: Conservative Test-Time Adaptation for Offline Policy Optimization

**Authors:** Jinxin Liu, Hongyin Zhang, Zifeng Zhuang, Yachen Kang, Donglin Wang, Bin Wang

### Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models

**Authors:** Yichao Cao, Qingfei Tang, Xiu Su, Song Chen, Shan You, Xiaobo Lu, Chang Xu

### Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models

**Authors:** Simian Luo, Chuanhao Yan, Chenxu Hu, Hang Zhao

### DiffComplete: Diffusion-based Generative 3D Shape Completion

**Authors:** Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Niessner, Chi-Wing Fu, Jiaya Jia

### DiffVL: Scaling Up Soft Body Manipulation using Vision-Language Driven Differentiable Physics

**Authors:** Zhiao Huang, Feng Chen, Yewen Pu, Chunru Lin, Hao Su, Chuang Gan

### Differentiable Clustering with Perturbed Spanning Forests

**Authors:** Lawrence Stewart, Francis Bach, Felipe Llinares-Lopez, Quentin Berthet

### Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning

**Authors:** Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li

### Diffusion Schrödinger Bridge Matching

**Authors:** Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet

### Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision

**Authors:** Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Josh Tenenbaum, Fredo Durand, Bill Freeman, Vincent Sitzmann

### Direct Training of SNN using Local Zeroth Order Method

**Authors:** Bhaskar Mukhoty, Velibor Bojkovic, William de Vazelhes, Xiaohan Zhao, Giulia De Masi, Huan Xiong, Bin Gu

### Directional diffusion models for graph representation learning

**Authors:** Run Yang, Yuling Yang, Fan Zhou, Qiang Sun

### Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design

**Authors:** Matthew T Jackson, Minqi Jiang, Jack Parker-Holder, Risto Vuorio, Chris Lu, Greg Farquhar, Shimon Whiteson, Jakob Foerster

### Discriminative Feature Attributions: Bridging Post Hoc Explainability and Inherent Interpretability

**Authors:** Usha Bhalla, Suraj Srinivas, Himabindu Lakkaraju

### Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models

**Authors:** Andy Zhou, Jindong Wang, Yu-Xiong Wang, Haohan Wang

### Distributional Pareto-Optimal Multi-Objective Reinforcement Learning

**Authors:** Xin-Qiang Cai, Pushi Zhang, Li Zhao, Jiang Bian, Masashi Sugiyama, Ashley Llorens

### Distributional Policy Evaluation: a Maximum Entropy approach to Representation Learning

**Authors:** Riccardo Zamboni, Alberto Maria Metelli, Marcello Restelli

### Diversify \& Conquer: Outcome-directed Curriculum RL via Out-of-Distribution Disagreement

**Authors:** Daesol Cho, Seungjae Lee, H. Jin Kim

### Do SSL Models Have Déjà Vu? A Case of Unintended Memorization in Self-supervised Learning

**Authors:** Casey Meehan, Florian Bordes, Pascal Vincent, Kamalika Chaudhuri, Chuan Guo

### Domain Re-Modulation for Few-Shot Generative Domain Adaptation

**Authors:** Yi Wu, Ziqiang Li, Chaoyue Wang, Heliang Zheng, Shanshan Zhao, Bin Li, Dacheng Tao

### Don't be so Monotone: Relaxing Stochastic Line Search in Over-Parameterized Models

**Authors:** Leonardo Galli, Holger Rauhut, Mark Schmidt

### Double Randomized Underdamped Langevin with Dimension-Independent Convergence Guarantee

**Authors:** Yuanshi Liu, Cong Fang, Tong Zhang

### Doubly Robust Augmented Transfer for Meta-Reinforcement Learning

**Authors:** Yuankun Jiang, Nuowen Kan, Chenglin Li, Wenrui Dai, Junni Zou, Hongkai Xiong

### Doubly-Robust Self-Training

**Authors:** Banghua Zhu, Mingyu Ding, Philip Jacobson, Ming Wu, Wei Zhan, Michael Jordan, Jiantao Jiao

### DreamHuman: Animatable 3D Avatars from Text

**Authors:** Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Bazavan, Mihai Fieraru, Cristian Sminchisescu

### DrugCLIP: Contrasive Protein-Molecule Representation Learning for Virtual Screening

**Authors:** Bowen Gao, Bo Qiang, Haichuan Tan, Yinjun Jia, Minsi Ren, Minsi Lu, Jingjing Liu, Wei-Ying Ma, Yanyan Lan

### DynPoint: Dynamic Neural Point For View Synthesis

**Authors:** Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu, Yiyuan Yang, Andrew Markham, Niki Trigoni

### [Spotlight] Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks

**Authors:** Blake Bordelon, Cengiz Pehlevan

### DäRF: Boosting Radiance Fields from Sparse Input Views with Monocular Depth Adaptation

**Authors:** Jiuhn Song, Seonghoon Park, Honggyu An, Seokju Cho, Min-Seop Kwak, Sungjin Cho, Seungryong Kim

### ELDEN: Exploration via Local Dependencies

**Authors:** Zizhao Wang, Jiaheng Hu, Peter Stone, Roberto Martín-Martín

### ESSEN: Improving Evolution State Estimation for Temporal Networks using Von Neumann Entropy

**Authors:** Qiyao Huang, Yingyue Zhang, Zhihong Zhang, Edwin Hancock

### EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras

**Authors:** Guangrong Zhao, Yurun Yang, Jingwei Liu, Ning Chen, Yiran Shen, Hongkai Wen, Guohao Lan

### Easy Learning from Label Proportions

**Authors:** Róbert Busa-Fekete, Heejin Choi, Travis Dick, Claudio Gentile, Andres Munoz Medina

### Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection

**Authors:** Xilie Xu, Jingfeng ZHANG, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

### Efficient Data Subset Selection to Generalize Training Across Models: Transductive and Inductive Networks

**Authors:** Eeshaan Jain, Tushar Nandy, Gaurav Aggarwal, Ashish Tendulkar, Rishabh Iyer, Abir De

### Efficient Hyper-parameter Optimization with Cubic Regularization

**Authors:** Zhenqian Shen, Hansi Yang, Yong Li, James Kwok, Quanming Yao

### Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models

**Authors:** Anant Raj, Umut Simsekli, Alessandro Rudi

### Efficient Testable Learning of Halfspaces with Adversarial Label Noise

**Authors:** Ilias Diakonikolas, Daniel Kane, Vasilis Kontonis, Sihan Liu, Nikos Zarifis

### [Oral] EgoEnv: Human-centric environment representations from egocentric video

**Authors:** Tushar Nagarajan, Santhosh Kumar Ramakrishnan, Ruta Desai, James Hillis, Kristen Grauman

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5D

### Eliminating Domain Bias for Federated Learning in Representation Space

**Authors:** Jianqing Zhang, Yang Hua, Jian Cao, Hao Wang, Tao Song, Zhengui XUE, Ruhui Ma, Haibing Guan

### Emergent Communication for Rules Reasoning

**Authors:** Yuxuan Guo, Yifan Hao, Rui Zhang, Enshuai Zhou, Zidong Du, xishan zhang, Xinkai Song, Yuanbo Wen, Yongwei Zhao, Xuehai Zhou, Jiaming Guo, Qi Yi, Shaohui Peng, Di Huang, Ruizhi Chen, Qi Guo, Yunji Chen

### End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes

**Authors:** Alexandre Maraval, Matthieu Zimmer, Antoine Grosnit, Haitham Bou Ammar

### Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models

**Authors:** Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, Jong Chul Ye

### Energy-based learning algorithms for analog computing: a comparative study

**Authors:** Benjamin Scellier, Maxence Ernoult, Jack Kendall, Suhas Kumar

### Enhancing Robot Program Synthesis Through Environmental Context

**Authors:** Tianyi Chen, Qidi Wang, Zhen Dong, Liwei Shen, Xin Peng

### Entropy-based Training Methods for Scalable Neural Implicit Samplers

**Authors:** Weijian Luo, Boya Zhang, Zhihua Zhang

### Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization

**Authors:** Haonan Yuan, Qingyun Sun, Xingcheng Fu, Ziwei Zhang, Cheng Ji, Hao Peng, Jianxin Li

### Equivariant Adaptation of Large Pretrained Models

**Authors:** Arnab Kumar Mondal, Siba Smarak Panigrahi, Oumar Kaba, Sai Rajeswar Mudumba, Siamak Ravanbakhsh

### Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation

**Authors:** Yuxuan Song, Jingjing Gong, Minkai Xu, Ziyao Cao, Yanyan Lan, Stefano Ermon, Hao Zhou, Wei-Ying Ma

### [Spotlight] Error Bounds for Learning with Vector-Valued Random Features

**Authors:** Samuel Lanthaler, Nicholas H. Nelsen

### Ess-InfoGAIL: Semi-supervised Imitation Learning from Imbalanced Demonstrations

**Authors:** Huiqiao Fu, Kaiqiang Tang, Yuanyang Lu, Yiming Qi, Guizhou Deng, Flood Sung, Chunlin Chen

### [Oral] Ethical Considerations for Responsible Data Curation

**Authors:** Jerone Andrews, Dora Zhao, William Thong, Apostolos Modas, Orestis Papakyriakopoulos, Alice Xiang

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5B

### [Oral] Evaluating Post-hoc Explanations for Graph Neural Networks via Robustness Analysis

**Authors:** Junfeng Fang, Wei Liu, Yuan Gao, Zemin Liu, An Zhang, Xiang Wang, Xiangnan He

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5A

### Evaluating Robustness and Uncertainty of Graph Models Under Structural Distributional Shifts

**Authors:** Gleb Bazhenov, Denis Kuznedelev, Andrey Malinin, Artem Babenko, Liudmila Prokhorenkova

### EvoPrompting: Language Models for Code-Level Neural Architecture Search

**Authors:** Angelica Chen, David Dohan, David So

### Exact Generalization Guarantees for (Regularized) Wasserstein Distributionally Robust Models

**Authors:** Waïss Azizian, Franck Iutzeler, Jérôme Malick

### [Spotlight] Explore In-Context Learning for 3D Point Cloud Understanding

**Authors:** Zhongbin Fang, Xiangtai Li, Xia Li, Joachim M Buhmann, Chen Change Loy, Mengyuan Liu

### Extensible Prompts for Language Models on Zero-shot Language Style Customization

**Authors:** Tao Ge, Hu Jing, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei

### FGPrompt: Fine-grained Goal Prompting for Image-goal Navigation

**Authors:** Xinyu Sun, Peihao Chen, Jugang Fan, Jian Chen, Thomas Li, Mingkui Tan

### FIND: A Function Description Benchmark for Evaluating Interpretability Methods

**Authors:** Sarah Schwettmann, Tamar Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, Antonio Torralba

### FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout

**Authors:** Irene Wang, Prashant Nair, Divya Mahajan

### FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space

**Authors:** Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang, Jinyang Li, Suhas Diggavi, Mani Srivastava, Tarek Abdelzaher

### False Discovery Proportion control for aggregated Knockoffs

**Authors:** Alexandre Blain, Bertrand Thirion, Olivier Grisel, Pierre Neuvial

### Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics

**Authors:** Guillaume Mahey, Laetitia Chapel, Gilles Gasso, Clément Bonet, Nicolas Courty

### Fast Partitioned Learned Bloom Filter

**Authors:** Atsuki Sato, Yusuke Matsui

### Faster Differentially Private Convex Optimization via Second-Order Methods

**Authors:** Arun Ganesh, Mahdi Haghifam, Thomas Steinke, Abhradeep Guha Thakurta

### Feature Likelihood Score: Evaluating the Generalization of Generative Models Using Samples

**Authors:** Marco Jiralerspong, Joey Bose, Ian Gemp, Chongli Qin, Yoram Bachrach, Gauthier Gidel

### FedFed: Feature Distillation against Data Heterogeneity in Federated Learning

**Authors:** Zhiqin Yang, Yonggang Zhang, Yu Zheng, Xinmei Tian, Hao Peng, Tongliang Liu, Bo Han

### FedGCN: Convergence-Communication Tradeoffs in Federated Training of Graph Convolutional Networks

**Authors:** Yuhang Yao, Weizhao Jin, Srivatsan Ravi, Carlee Joe-Wong

### FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning

**Authors:** Jinyuan Jia, Zhuowen Yuan, Dinuka Sahabandu, Luyao Niu, Arezoo Rajabi, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

### Finding Safe Zones of Markov Decision Processes Policies

**Authors:** Lee Cohen, Yishay Mansour, Michal Moshkovitz

### Fine-Grained Visual Prompting

**Authors:** Lingfeng Yang, Yueze Wang, Xiang Li, Xinlong Wang, Jian Yang

### Flat Seeking Bayesian Neural Networks

**Authors:** Van-Anh Nguyen, Tung-Long Vuong, Hoang Phan, Thanh-Toan Do, Dinh Phung, Trung Le

### Flexible Attention-Based Multi-Policy Fusion for Efficient Deep Reinforcement Learning

**Authors:** Zih-Yun Chiu, Yi-Lin Tuan, William Yang Wang, Michael Yip

### Flow-Attention-based Spatio-Temporal Aggregation Network for 3D Mask Detection

**Authors:** Yuxin Cao, Yian Li, Yumeng Zhu, Derui Wang, Minhui Xue

### Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection

**Authors:** Haibao Yu, Yingjuan Tang, Enze Xie, Jilei Mao, Ping Luo, Zaiqing Nie

### FlowPG: Action-constrained Policy Gradient with Normalizing Flows

**Authors:** Janaka Brahmanage, Jiajing LING, Akshat Kumar

### Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation

**Authors:** Yuan Wang, Naisong Luo, Tianzhu Zhang

### Focused Transformer: Contrastive Training for Context Scaling

**Authors:** Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, Piotr Miłoś

### [Spotlight] Follow-ups Also Matter: Improving Contextual Bandits via Post-serving Contexts

**Authors:** Chaoqi Wang, Ziyu Ye, Zhe Feng, Haifeng Xu

### Formalizing locality for normative synaptic plasticity models

**Authors:** Colin Bredenberg, Ezekiel Williams, Cristina Savin, Blake Richards, Guillaume Lajoie

### FouriDown: Factoring Down-Sampling into Shuffling and Superposing

**Authors:** Qi Zhu, man zhou, Jie Huang, Naishan Zheng, Hongzhi Gao, Chongyi Li, Yuan Xu, Feng Zhao

### Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator

**Authors:** Hanzhuo Huang, Yufan Feng, Cheng Shi, Lan Xu, Jingyi Yu, Sibei Yang

### Frequency-Enhanced Data Augmentation for Vision-and-Language Navigation

**Authors:** Keji He, Chenyang Si, Zhihe Lu, Yan Huang, Liang Wang, Xinchao Wang

### Frequency-domain MLPs are More Effective Learners in Time Series Forecasting

**Authors:** Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, Zhendong Niu

### From Trainable Negative Depth to Edge Heterophily in Graphs

**Authors:** Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Minghua Xu, Mahashweta Das, Hao Yang, Hanghang Tong

### From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models

**Authors:** Roy Uziel, Or Dinari, Oren Freifeld

### Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge

**Authors:** Abhin Shah, Karthikeyan Shanmugam, Murat Kocaoglu

### Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation and Elaboration

**Authors:** Haitao Lin, Yufei Huang, Odin Zhang, Yunfan Liu, Lirong Wu, Siyuan Li, Zhiyuan Chen, Stan Z. Li

### GALOPA: Graph Transport Learning with Optimal Plan Alignment

**Authors:** Yejiang Wang, Yuhai Zhao, Daniel Zhengkui Wang, Ling Li

### GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels

**Authors:** Xin Zheng, Miao Zhang, Chunyang Chen, Soheila Molaei, Chuan Zhou, Shirui Pan

### GPEX, A Framework For Interpreting Artificial Neural Networks

**Authors:** Gilbert Bigras, Nilanjan Ray

### GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

**Authors:** Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang

### Game Solving with Online Fine-Tuning

**Authors:** Ti-Rong Wu, Hung Guei, Ting Han Wei, Chung-Chin Shih, Jui-Te Chin, I-Chen Wu

### Generalization bounds for neural ordinary differential equations and deep residual networks

**Authors:** Pierre Marion

### Generalized Belief Transport

**Authors:** Junqi Wang, PEI WANG, Patrick Shafto

### Generalized test utilities for long-tail performance in extreme multi-label classification

**Authors:** Erik Schultheis, Marek Wydmuch, Wojciech Kotlowski, Rohit Babbar, Krzysztof Dembczynski

### Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion

**Authors:** Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, Xiangnan He

### Generating Behaviorally Diverse Policies with Latent Diffusion Models

**Authors:** Shashank Hegde, Sumeet Batra, K.R. Zentner, Gaurav Sukhatme

### Generative Category-level Object Pose Estimation via Diffusion Models

**Authors:** Jiyao Zhang, Mingdong Wu, Hao Dong

### Generative Pre-Training of Spatio-Temporal Graph Neural Networks

**Authors:** Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang

### Generator Born from Classifier

**Authors:** Runpeng Yu, Xinchao Wang

### Geodesic Multi-Modal Mixup for Robust Fine-Tuning

**Authors:** Changdae Oh, Junhyuk So, Hoyoon Byun, YongTaek Lim, Minchul Shin, Jong-June Jeon, Kyungwoo Song

### Geometry-Aware Adaptation for Pretrained Models

**Authors:** Nicholas Roberts, Xintong Li, Dyah Adila, Sonia Cromp, Tzu-Heng Huang, Jitian Zhao, Frederic Sala

### Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design

**Authors:** Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer

### Global Convergence Analysis of Local SGD for Two-layer Neural Network without Overparameterization

**Authors:** Yajie Bao, Amarda Shehu, Mingrui Liu

### Global Update Tracking: A Decentralized Learning Algorithm for Heterogeneous Data

**Authors:** Sai Aparna Aketi, Abolfazl Hashemi, Kaushik Roy

### Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction

**Authors:** Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, Yi Yang

### [Oral] Going beyond persistent homology using persistent homology

**Authors:** Johanna Immonen, Amauri Souza, Vikas Garg

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5A

### Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

**Authors:** Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, Kai Han

### Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians

**Authors:** Rainer Engelken

### Gradient Informed Proximal Policy Optimization

**Authors:** Sanghyun Son, Laura Zheng, Ryan Sullivan, Yi-Ling Qiao, Ming Lin

### Graph-Structured Gaussian Processes for Transferable Graph Learning

**Authors:** Jun Wu, Lisa Ainsworth, Andrew Leakey, Haixun Wang, Jingrui He

### GraphPatcher: Mitigating Degree Bias for Graph Neural Networks via Test-time Augmentation

**Authors:** Mingxuan Ju, Tong Zhao, Wenhao Yu, Neil Shah, Yanfang Ye

### [Spotlight] Grounding Neural Inference with Satisfiability Modulo Theories

**Authors:** Zifan Wang, Saranya Vijayakumar, Kaiji Lu, Vijay Ganesh, Somesh Jha, Matt Fredrikson

### H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation

**Authors:** Yanjie Ze, Yuyao Liu, Ruizhe Shi, Jiaxin Qin, Zhecheng Yuan, Jiashun Wang, Huazhe Xu

### Hierarchical Gaussian Mixture based Task Generative Model for Robust Meta-Learning

**Authors:** Yizhou Zhang, Jingchao Ni, Wei Cheng, Zhengzhang Chen, Liang Tong, Haifeng Chen, Yan Liu

### [Spotlight] High-Fidelity Audio Compression with Improved RVQGAN

**Authors:** Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar

### History Filtering in Imperfect Information Games: Algorithms and Complexity

**Authors:** Christopher Solinas, Doug Rebstock, Nathan Sturtevant, Michael Buro

### Homotopy-based training of NeuralODEs for accurate dynamics discovery

**Authors:** Joon-Hyuk Ko, Hankyul Koh, Nojun Park, Wonho Jhe

### How Does Adaptive Optimization Impact Local Neural Network Geometry?

**Authors:** Kaiqi Jiang, Dhruv Malik, Yuanzhi Li

### How do Minimum-Norm Shallow Denoisers Look in Function Space?

**Authors:** Chen Zeno, Greg Ongie, Yaniv Blumenfeld, Nir Weinberger, Daniel Soudry

### How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model

**Authors:** Michael Hanna, Ollie Liu, Alexandre Variengien

### How to Select Which Active Learning Strategy is Best Suited for Your Specific Problem and Budget

**Authors:** Guy Hacohen, Daphna Weinshall

### [Spotlight] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution

**Authors:** Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, Stefano Ermon, Christopher Ré, Stephen Baccus

### IEBins: Iterative Elastic Bins for Monocular Depth Estimation

**Authors:** Shuwei Shao, Zhongcai Pei, Xingming Wu, Zhong Liu, Weihai Chen, Zhengguo Li

### ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns

**Authors:** Ren Li, Benoit Guillard, Pascal Fua

### Identifiability Guarantees for Causal Disentanglement from Soft Interventions

**Authors:** Jiaqi Zhang, Kristjan Greenewald, Chandler Squires, Akash Srivastava, Karthikeyan Shanmugam, Caroline Uhler

### Imagine That! Abstract-to-Intricate Text-to-Image Synthesis with Scene Graph Hallucination Diffusion

**Authors:** Shengqiong Wu, Hao Fei, Hanwang Zhang, Tat-Seng Chua

### Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis

**Authors:** Zhu Wang, Sourav Medya, Sathya Ravi

### [Spotlight] Improved Convergence in High Probability of Clipped Gradient Methods with Heavy Tailed Noise

**Authors:** Ta Duy Nguyen, Thien H Nguyen, Alina Ene, Huy Nguyen

### Improving Compositional Generalization using Iterated Learning and Simplicial Embeddings

**Authors:** Yi Ren, Samuel Lavoie, Michael Galkin, Danica J. Sutherland, Aaron Courville

### Improving Language Plasticity via Pretraining with Active Forgetting

**Authors:** Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Adelani, Sebastian Riedel, Mikel Artetxe

### In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer

**Authors:** Yuzhou Cao, Hussein Mozannar, Lei Feng, Hongxin Wei, Bo An

### Incentivizing Honesty among Competitors in Collaborative Learning and Optimization

**Authors:** Florian E. Dorner, Nikola Konstantinov, Georgi Pashaliev, Martin Vechev

### [Spotlight] Individual Arbitrariness and Group Fairness

**Authors:** Carol Long, Hsiang Hsu, Wael Alghamdi, Flavio Calmon

### Individualized Dosing Dynamics via Neural Eigen Decomposition

**Authors:** Stav Belogolovsky, Ido Greenberg, Danny Eytan, Shie Mannor

### Inferring Hybrid Neural Fluid Fields from Videos

**Authors:** Hong-Xing Yu, Yang Zheng, Yuan Gao, Yitong Deng, Bo Zhu, Jiajun Wu

### Inferring the Future by Imagining the Past

**Authors:** Kartik Chandra, Tony Chen, Tzu-Mao Li, Jonathan Ragan-Kelley, Josh Tenenbaum

### InfoPrompt: Information-Theoretic Soft Prompt Tuning for Natural Language Understanding

**Authors:** Junda Wu, Tong Yu, Rui Wang, Zhao Song, Ruiyi Zhang, Handong Zhao, Chaochao Lu, Shuai Li, Ricardo Henao

### Initialization-Dependent Sample Complexity of Linear Predictors and Neural Networks

**Authors:** Roey Magen, Ohad Shamir

### InsActor: Instruction-driven Physics-based Characters

**Authors:** Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Xiao Ma, Liang Pan, Ziwei Liu

### Interpretable Prototype-based Graph Information Bottleneck

**Authors:** Sangwoo Seo, Sungwon Kim, Chanyoung Park

### Inverse Dynamics Pretraining Learns Good Representations for Multitask Imitation

**Authors:** David Brandfonbrener, Ofir Nachum, Joan Bruna

### Inverse Preference Learning: Preference-based RL without a Reward Function

**Authors:** Joey Hejna, Dorsa Sadigh

### Is Heterogeneity Notorious? Taming Heterogeneity to Handle Test-Time Shift in Federated Learning

**Authors:** Yue Tan, Chen Chen, Weiming Zhuang, Xin Dong, Lingjuan Lyu, Guodong Long

### Iterative Reachability Estimation for Safe Reinforcement Learning

**Authors:** Milan Ganai, Zheng Gong, Chenning Yu, Sylvia Herbert, Sicun Gao

### Joint Feature and Differentiable $ k $-NN Graph Learning using Dirichlet Energy

**Authors:** Lei Xu, Lei Chen, Rong Wang, Feiping Nie, Xuelong Li

### K-Nearest-Neighbor Local Sampling Based Conditional Independence Testing

**Authors:** Shuai Li, Yingjie Zhang, Hongtu Zhu, Christina Wang, Hai Shu, Ziqi Chen, Zhuoran Sun, Yanfeng Yang

### Kernel-Based Tests for Likelihood-Free Hypothesis Testing

**Authors:** Patrik Robert Gerber, Tianze Jiang, Yury Polyanskiy, Rui Sun

### [Spotlight] Kernelized Cumulants: Beyond Kernel Mean Embeddings

**Authors:** Patric Bonnier, Harald Oberhauser, Zoltan Szabo

### Knowledge Distillation for High Dimensional Search Index

**Authors:** Zepu Lu, Jin Chen, Defu Lian, ZAIXI ZHANG, Yong Ge, Enhong Chen

### L2T-DLN: Learning to Teach with Dynamic Loss Network

**Authors:** Zhaoyang Hai, Liyuan Pan, Xiabi Liu, Zhengzheng Liu, Mirna Yunita

### LART: Neural Correspondence Learning with Latent Regularization Transformer for 3D Motion Transfer

**Authors:** Haoyu Chen, Hao Tang, Radu Timofte, Luc V Gool, Guoying Zhao

### LEPARD: Learning Explicit Part Discovery for 3D Articulated Shape Reconstruction

**Authors:** Di Liu, Anastasis Stathopoulos, Qilong Zhangli, Yunhe Gao, Dimitris Metaxas

### LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections

**Authors:** Muhammad Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Mateusz Kozinski, Rogerio Feris, Horst Bischof

### Label Correction of Crowdsourced Noisy Annotations with an Instance-Dependent Noise Transition Model

**Authors:** Hui GUO, Boyu Wang, Grace Yi

### Label Poisoning is All You Need

**Authors:** Rishi Jha, Jonathan Hayase, Sewoong Oh

### Label-Retrieval-Augmented Diffusion Models for Learning from Noisy Labels

**Authors:** Jian Chen, Ruiyi Zhang, Tong Yu, Rohan Sharma, Zhiqiang Xu, Tong Sun, Changyou Chen

### Labeling Neural Representations with Inverse Recognition

**Authors:** Kirill Bykov, Laura Kopf, Shinichi Nakajima, Marius Kloft, Marina Höhne

### LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas

**Authors:** Kensen Shi, Hanjun Dai, Wen-Ding Li, Kevin Ellis, Charles Sutton

### Landscape Surrogate: Learning Decision Losses for Mathematical Optimization Under Partial Information

**Authors:** Arman Zharmagambetov, Brandon Amos, Aaron Ferber, Taoan Huang, Bistra Dilkina, Yuandong Tian

### Language-based Action Concept Spaces Improve Video Self-Supervised Learning

**Authors:** Kanchana Ranasinghe, Michael S Ryoo

### Large Language Models Are Zero-Shot Time Series Forecasters

**Authors:** Nate Gruver, Marc Finzi, Shikai Qiu, Andrew Wilson

### Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language.

**Authors:** Eghbal Hosseini, Evelina Fedorenko

### Latent Space Translation via Semantic Alignment

**Authors:** Valentino Maiorca, Luca Moschella, Antonio Norelli, Marco Fumero, Francesco Locatello, Emanuele Rodolà

### Learn to Categorize or Categorize to Learn? Self-Coding for Generalized Category Discovery

**Authors:** Sarah Rastegar, Hazel Doughty, Cees Snoek

### Learning Cuts via Enumeration Oracles

**Authors:** Daniel Thuerck, Boro Sofranac, Marc E Pfetsch, Sebastian Pokutta

### Learning Dictionary for Visual Attention

**Authors:** Yingjie Liu, Xuan Liu, Hui Yu, XUAN TANG, Xian Wei

### Learning Efficient Coding of Natural Images with Maximum Manifold Capacity Representations

**Authors:** Thomas Yerxa, Yilun Kuang, Eero Simoncelli, SueYeon Chung

### Learning Energy-Based Prior Model with Diffusion-Amortized MCMC

**Authors:** Peiyu Yu, Yaxuan Zhu, Sirui Xie, Xiaojian (Shawn) Ma, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu

### Learning Invariant Molecular Representation in Latent Discrete Space

**Authors:** Xiang Zhuang, Qiang Zhang, Keyan Ding, Yatao Bian, Xiao Wang, Jingsong Lv, Hongyang Chen, Huajun Chen

### Learning List-Level Domain-Invariant Representations for Ranking

**Authors:** Ruicheng Xian, Honglei Zhuang, Zhen Qin, Hamed Zamani, Jing Lu, Ji Ma, Kai Hui, Han Zhao, Xuanhui Wang, Michael Bendersky

### Learning Modulated Transformation in GANs

**Authors:** Ceyuan Yang, Qihang Zhang, Yinghao Xu, Jiapeng Zhu, Yujun Shen, Bo Dai

### Learning Motion Refinement for Unsupervised Face Animation

**Authors:** Jiale Tao, Shuhang Gu, Wen Li, Lixin Duan

### Learning Probabilistic Symmetrization for Architecture Agnostic Equivariance

**Authors:** Jinwoo Kim, Dat Nguyen, Ayhan Suleymanzade, Hyeokjun An, Seunghoon Hong

### Learning Provably Robust Estimators for Inverse Problems via Jittering

**Authors:** Anselm Krainovic, Mahdi Soltanolkotabi, Reinhard Heckel

### Learning Re-sampling Methods with Parameter Attribution for Image Super-resolution

**Authors:** Xiaotong Luo, Yuan Xie, Yanyun Qu

### Learning Reliable Logical Rules with SATNet

**Authors:** Zhaoyu Li, Jinpei Guo, Yuhe Jiang, Xujie Si

### Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer

**Authors:** Jianwei Zhang, Suren Jayasuriya, Visar Berisha

### Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping

**Authors:** Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, Hao Dong

### Learning To Dive In Branch And Bound

**Authors:** Max Paulus, Andreas Krause

### Learning Topology-Agnostic EEG Representations with Geometry-Aware Modeling

**Authors:** Ke Yi, Yansen Wang, Kan Ren, Dongsheng Li

### Learning and processing the ordinal information of temporal sequences in recurrent neural circuits

**Authors:** xiaolong zou, Zhikun Chu, Qinghai Guo, Jie Cheng, Bo Ho, Si Wu, Yuanyuan Mi

### Learning from Both Structural and Textual Knowledge for Inductive Knowledge Graph Completion

**Authors:** Kunxun Qi, Jianfeng Du, Hai Wan

### Learning to Compress Prompts with Gist Tokens

**Authors:** Jesse Mu, Xiang Li, Noah Goodman

### Learning to Configure Separators in Branch-and-Cut

**Authors:** Sirui Li, Wenbin Ouyang, Max Paulus, Cathy Wu

### Learning to Group Auxiliary Datasets for Molecule

**Authors:** Tinglin Huang, Ziniu Hu, Rex Ying

### Learning to Influence Human Behavior with Offline Reinforcement Learning

**Authors:** Joey Hong, Sergey Levine, Anca Dragan

### Learning with Explanation Constraints

**Authors:** Rattana Pukdee, Dylan Sam, J. Zico Kolter, Maria-Florina Balcan, Pradeep Ravikumar

### Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection

**Authors:** Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li

### Leveraging the two-timescale regime to demonstrate convergence of neural networks

**Authors:** Pierre Marion, Raphaël Berthier

### Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory

**Authors:** Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, Rui Yan

### Limits, approximation and size transferability for GNNs on sparse graphs via graphops

**Authors:** Thien Le, Stefanie Jegelka

### Linear Time Algorithms for k-means with Multi-Swap Local Search

**Authors:** Junyu Huang, Qilong Feng, Ziyun Huang, Jinhui Xu, Jianxin Wang

### [Spotlight] Locality Sensitive Hashing in Fourier Frequency Domain For Soft Set Containment Search

**Authors:** Indradyumna Roy, Rishi Agarwal, Soumen Chakrabarti, Anirban Dasgupta, Abir De

### Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline RL

**Authors:** Peng Cheng, Xianyuan Zhan, zhihao wu, Wenjia Zhang, Youfang Lin, Shou cheng Song, Han Wang, Li Jiang

### Lookaround Optimizer: $k$ steps around, 1 step average

**Authors:** Jiangtao Zhang, Shunyu Liu, Jie Song, Tongtian Zhu, Zhengqi Xu, Mingli Song

### Lower Bounds on Adaptive Sensing for Matrix Recovery

**Authors:** Praneeth Kacham, David Woodruff

### MIMONets: Multiple-Input-Multiple-Output Neural Networks Exploiting Computation in Superposition

**Authors:** Nicolas Menet, Michael Hersche, Geethan Karunaratne, Luca Benini, Abu Sebastian, Abbas Rahimi

### MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

**Authors:** Mohammad Mozaffari, Sikan Li, Zhao Zhang, Maryam Mehri Dehnavi

### Macro Placement by Wire-Mask-Guided Black-Box Optimization

**Authors:** Yunqi Shi, Ke Xue, Song Lei, Chao Qian

### Managing Temporal Resolution in Continuous Value Estimation: A Fundamental Trade-off

**Authors:** Zichen Zhang, Johannes Kirschner, Junxi Zhang, Francesco Zanini, Alex Ayoub, Masood Dehghan, Dale Schuurmans

### Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits

**Authors:** Muhammad Faaiz Taufiq, Arnaud Doucet, Rob Cornish, Jean-Francois Ton

### [Spotlight] Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction

**Authors:** Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, Huaping Liu

### Maximum Independent Set: Self-Training through Dynamic Programming

**Authors:** Lorenzo Brusca, Lars C.P.M. Quaedvlieg, Stratis Skoulakis, Grigorios Chrysos, Volkan Cevher

### Meek Separators and Their Applications in Targeted Causal Discovery

**Authors:** Kirankumar Shiragur, Jiaqi Zhang, Caroline Uhler

### [Spotlight] Memory Efficient Optimizers with 4-bit States

**Authors:** Bingrui Li, Jianfei Chen, Jun Zhu

### Memory-Constrained Algorithms for Convex Optimization

**Authors:** Moise Blanchard, Junhui Zhang, Patrick Jaillet

### Meta-Learning Adversarial Bandit Algorithms

**Authors:** Misha Khodak, Ilya Osadchiy, Keegan Harris, Maria-Florina Balcan, Kfir Y. Levy, Ron Meir, Steven Wu

### Metropolis Sampling for Constrained Diffusion Models

**Authors:** Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, Valentin De Bortoli

### [Spotlight] Minimum-Risk Recalibration of Classifiers

**Authors:** Zeyu Sun, Dogyoon Song, Alfred Hero

### Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser

**Authors:** Yung-Hsuan Lai, Yen-Chun Chen, Frank Wang

### [Spotlight] Model Spider: Learning to Rank Pre-Trained Models Efficiently

**Authors:** Yi-Kai Zhang, Ting-Ji Huang, Yao-Xiang Ding, De-Chuan Zhan, Han-Jia Ye

### Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context

**Authors:** Lakshya A Agrawal, Aditya Kanade, Navin Goyal, Shuvendu Lahiri, Sriram Rajamani

### Moral Responsibility for AI Systems

**Authors:** Sander Beckers

### Most Neural Networks Are Almost Learnable

**Authors:** Amit Daniely, Nati Srebro, Gal Vardi

### Multi Time Scale World Models

**Authors:** Vaisakh Shaj Kumar, SALEH GHOLAM ZADEH, Ozan Demir, Luiz Douat, Gerhard Neumann

### Multi-Agent Meta-Reinforcement Learning: Sharper Convergence Rates with Task Similarity

**Authors:** Weichao Mao, Haoran Qiu, Chen Wang, Hubertus Franke, Zbigniew Kalbarczyk, Ravishankar Iyer, Tamer Basar

### Multi-Fidelity Multi-Armed Bandits Revisited

**Authors:** Xuchuang Wang, Qingyun Wu, Wei Chen, John C.S. Lui

### Multi-Head Adapter Routing for Cross-Task Generalization

**Authors:** Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, Alessandro Sordoni

### Multi-Objective Intrinsic Reward Learning for Conversational Recommender Systems

**Authors:** Zhendong Chu, Nan Wang, Hongning Wang

### Multi-modal Queried Object Detection in the Wild

**Authors:** Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, Changsheng Xu

### Multi-resolution Spectral Coherence for Graph Generation with Score-based Diffusion

**Authors:** Hyuna Cho, Minjae Jeong, Sooyeon Jeon, Sungsoo Ahn, Won Hwa Kim

### Multi-task learning with summary statistics

**Authors:** Parker Knight, Rui Duan

### MultiMoDN—Multimodal, Multi-Task, Interpretable Modular Networks

**Authors:** Vinitra Swamy, Malika Satayeva, Jibril Frej, Thierry Bossy, Thijs Vogels, Martin Jaggi, Tanja Käser, Mary-Anne Hartley

### Multiclass Boosting: Simple and Intuitive Weak Learning Criteria

**Authors:** Nataly Brukhim, Amit Daniely, Yishay Mansour, Shay Moran

### NPCL: Neural Processes for Uncertainty-Aware Continual Learning

**Authors:** Saurav Jha, Dong Gong, He Zhao, Lina Yao

### NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos

**Authors:** Jinxi Li, Ziyang Song, Bo Yang

### NeRF Revisited: Fixing Quadrature Instability in Volume Rendering

**Authors:** Mikaela Angelina Uy, Kiyohiro Nakayama, Guandao Yang, Rahul Thomas, Leonidas Guibas, Ke Li

### Near-Optimal Algorithms for Gaussians with Huber Contamination: Mean Estimation and Linear Regression

**Authors:** Ilias Diakonikolas, Daniel Kane, Ankit Pensia, Thanasis Pittas

### Neural (Tangent Kernel) Collapse

**Authors:** Mariia Seleznova, Dana Weitzner, Raja Giryes, Gitta Kutyniok, Hung-Hsu Chou

### [Spotlight] Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes

**Authors:** Aran Nayebi, Rishi Rajalingham, Mehrdad Jazayeri, Guangyu Robert Yang

### Neural Harmonics: Bridging Spectral Embedding and Matrix Completion in Self-Supervised Learning

**Authors:** Marina Munkhoeva, Ivan Oseledets

### Neural Lighting Simulation for Urban Scenes

**Authors:** Ava Pun, Gary Sun, Jingkang Wang, Yun Chen, Ze Yang, Sivabalan Manivasagam, Wei-Chiu Ma, Raquel Urtasun

### Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data

**Authors:** Jang-Hyun Kim, Sangdoo Yun, Hyun Oh Song

### Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction

**Authors:** Chih-Yu (Andrew) Lai, Fan-Keng Sun, Zhengqi Gao, Jeffrey H Lang, Duane Boning

### Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization

**Authors:** Quanqi Hu, Dixian Zhu, Tianbao Yang

### Non-adversarial training of Neural SDEs with signature kernel scores

**Authors:** Zacharia Issa, Blanka Horvath, Maud Lemercier, Cristopher Salvi

### Norm-guided latent space exploration for text-to-image generation

**Authors:** Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, Gal Chechik

### Normalization Layers Are All That Sharpness-Aware Minimization Needs

**Authors:** Maximilian Mueller, Tiffany Vlaar, David Rolnick, Matthias Hein

### Not All Out-of-Distribution Data Are Harmful to Open-Set Active Learning

**Authors:** Yang Yang, Yuxuan Zhang, XIN SONG, Yi Xu

### Object-Centric Slot Diffusion

**Authors:** Jindong Jiang, Fei Deng, Gautam Singh, Sungjin Ahn

### Offline Minimax Soft-Q-learning Under Realizability and Partial Coverage

**Authors:** Masatoshi Uehara, Nathan Kallus, Jason Lee, Wen Sun

### On Certified Generalization in Structured Prediction

**Authors:** Bastian Boll, Christoph Schnörr

### [Spotlight] On Learning Necessary and Sufficient Causal Graphs

**Authors:** Hengrui Cai, Yixin Wang, Michael Jordan, Rui Song

### On Masked Pre-training and the Marginal Likelihood

**Authors:** Pablo Moreno-Muñoz, Pol Garcia Recasens, Søren Hauberg

### On Private and Robust Bandits

**Authors:** Yulian Wu, Xingyu Zhou, Youming Tao, Di Wang

### On permutation symmetries in Bayesian neural network posteriors: a variational perspective

**Authors:** Simone Rossi, Ankit Singh, Thomas Hannagan

### On the Consistency of Maximum Likelihood Estimation of Probabilistic Principal Component Analysis

**Authors:** Arghya Datta, Sayak Chakrabarty

### On the Identifiability of Sparse ICA without Assuming Non-Gaussianity

**Authors:** Ignavier Ng, Yujia Zheng, Xinshuai Dong, Kun Zhang

### On the Properties of Kullback-Leibler Divergence Between Multivariate Gaussian Distributions

**Authors:** Yufeng Zhang, Jialu Pan, Li Ken Li, Wanwei Liu, Zhenbang Chen, Xinwang Liu, J Wang

### On the Robustness of Mechanism Design under Total Variation Distance

**Authors:** Anuran Makur, Marios Mertzanidis, Alexandros Psomas, Athina Terzoglou

### On the Role of Noise in the Sample Complexity of Learning Recurrent Neural Networks: Exponential Gaps for Long Sequences

**Authors:** Alireza F. Pour, Hassan Ashtiani

### On the choice of Perception Loss Function for Learned Video Compression

**Authors:** Sadaf Salehkalaibar, Truong Buu Phan, Jun Chen, Wei Yu, Ashish Khisti

### On the impact of activation  and normalization in obtaining  isometric embeddings at initialization

**Authors:** Amir Joudaki, Hadi Daneshmand, Francis Bach

### [Spotlight] One Fits All: Power General Time Series Analysis by Pretrained LM

**Authors:** Tian Zhou, Peisong Niu, xue wang, Liang Sun, Rong Jin

### One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning

**Authors:** Marc Rigter, Bruno Lacerda, Nick Hawes

### [Spotlight] Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost

**Authors:** Yu-Jie Zhang, Masashi Sugiyama

### Online Ad Allocation with Predictions

**Authors:** Fabian Spaeh, Alina Ene

### Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations

**Authors:** Yiheng Lin, James A. Preiss, Emile Anand, Yingying Li, Yisong Yue, Adam Wierman

### Open Compound Domain Adaptation with Object Style Compensation for Semantic Segmentation

**Authors:** Tingliang Feng, Hao Shi, Xueyang Liu, Wei Feng, Liang Wan, Yanlin Zhou, Di Lin

### Open Visual Knowledge Extraction via Relation-Oriented Multimodality Model Prompting

**Authors:** Hejie Cui, Xinyu Fang, Zihan Zhang, Ran Xu, Xuan Kan, Xin Liu, Yue Yu, Manling Li, Yangqiu Song, Carl Yang

### [Spotlight] Optimal Exploration for Model-Based RL in Nonlinear Systems

**Authors:** Andrew Wagenmaker, Guanya Shi, Kevin Jamieson

### [Spotlight] Optimal Guarantees for Algorithmic Reproducibility and Gradient Complexity in Convex Optimization

**Authors:** Liang Zhang, Junchi YANG, Amin Karbasi, Niao He

### Optimality in Mean Estimation: Beyond Worst-Case, Beyond Sub-Gaussian, and Beyond $1+\alpha$ Moments

**Authors:** Trung Dang, Jasper Lee, Maoyuan 'Raymond' Song, Paul Valiant

### Optimistic Rates for Multi-Task Representation Learning

**Authors:** Austin Watkins, Enayat Ullah, Thanh Nguyen-Tang, Raman Arora

### [Oral] Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of Policy-Gradient Method

**Authors:** Constantine Caramanis, Dimitris Fotakis, Alkis Kalavasis, Vasilis Kontonis, Christos Tzamos

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5C

### PCF-GAN: generating sequential data via the characteristic function of measures on the path space

**Authors:** Hang Lou, Siran Li, Hao Ni

### PDF: Point Diffusion Implicit Function for Large-scale Scene Neural Representation

**Authors:** Yuhan Ding, Fukun Yin, Jiayuan Fan, Hui Li, Xin Chen, Wen Liu, Chongshan Lu, Gang Yu, Tao Chen

### PDP: Parameter-free Differentiable Pruning is All You Need

**Authors:** Minsik Cho, Saurabh Adya, Devang Naik

### PHOTOSWAP: Personalized Subject Swapping in Images

**Authors:** Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, HE Zhang, Jianming Zhang, HyunJoon Jung, Xin Eric Wang

### PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks

**Authors:** Ian Char, Jeff Schneider

### POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images

**Authors:** Antonin Vobecky, Oriane Siméoni, David Hurych, Spyridon Gidaris, Andrei Bursuc, Patrick Pérez, Josef Sivic

### PRED: Pre-training via Semantic Rendering on LiDAR Point Clouds

**Authors:** Hao Yang, Haiyang Wang, Di Dai, Liwei Wang

### Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies

**Authors:** Wei Fang, Zhaofei Yu, Zhaokun Zhou, Ding Chen, Yanqi Chen, Zhengyu Ma, Timothée Masquelier, Yonghong Tian

### [Spotlight] Parallel Submodular Function Minimization

**Authors:** Deeparnab Chakrabarty, Andrei Graur, Haotian Jiang, Aaron Sidford

### Parameter-efficient Tuning of Large-scale Multimodal Foundation Model

**Authors:** Haixin Wang, Xinlong Yang, Jianlong Chang, Dian Jin, Jinan Sun, Shikun Zhang, Xiao Luo, Qi Tian

### [Spotlight] Parsel🐍: Algorithmic Reasoning with Language Models by Composing Decompositions

**Authors:** Eric Zelikman, Qian Huang, Gabriel Poesia, Noah Goodman, Nick Haber

### Partial Matrix Completion

**Authors:** Elad Hazan, Adam Tauman Kalai, Varun Kanade, Clara Mohri, Y. Jennifer Sun

### Particle-based Variational Inference with Generalized Wasserstein Gradient Flow

**Authors:** Ziheng Cheng, Shiyue Zhang, Longlin Yu, Cheng Zhang

### Parts of Speech–Grounded Subspaces in Vision-Language Models

**Authors:** James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, Ioannis Patras

### Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models

**Authors:** Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang "Atlas" Wang, Weizhu Chen, Mingyuan Zhou

### Pengi: An Audio Language Model for Audio Tasks

**Authors:** Soham Deshmukh, Benjamin Elizalde, Rita Singh, Huaming Wang

### Performance Scaling via Optimal Transport: Enabling Data Selection from Partially Revealed Sources

**Authors:** Feiyang Kang, Hoang Anh Just, Anit Kumar Sahu, Ruoxi Jia

### Permutation Equivariant Neural Functionals

**Authors:** Allan Zhou, Kaien Yang, Kaylee Burns, Adriano Cardace, Yiding Jiang, Samuel Sokota, J. Zico Kolter, Chelsea Finn

### Point Cloud Completion with Pretrained Text-to-Image Diffusion Models

**Authors:** Yoni Kasten, Ohad Rahamim, Gal Chechik

### PointGPT: Auto-regressively Generative Pre-training from Point Clouds

**Authors:** Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue

### Policy Finetuning in Reinforcement Learning via Design of Experiments using Offline Data

**Authors:** Ruiqi Zhang, Andrea Zanette

### Post-processing Private Synthetic Data for Improving Utility on Selected Measures

**Authors:** Hao Wang, Shivchander Sudalairaj, John Henning, Kristjan Greenewald, Akash Srivastava

### Practical and Asymptotically Exact Conditional Sampling in Diffusion Models

**Authors:** Luhuan Wu, Brian Trippe, Christian Naesseth, John Cunningham, David Blei

### Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting

**Authors:** Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang (Bernie) Wang

### Predicting Global Label Relationship Matrix for Graph Neural Networks under Heterophily

**Authors:** Langzhang Liang, Xiangjing Hu, Zenglin Xu, Zixing Song, Irwin King

### [Spotlight] Prefix-Tree Decoding for Predicting Mass Spectra from Molecules

**Authors:** Samuel Goldman, John Bradshaw, Jiayi Xin, Connor Coley

### [Spotlight] Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision

**Authors:** Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan

### Principled Weight Initialisation for Input-Convex Neural Networks

**Authors:** Pieter-Jan Hoedt, Günter Klambauer

### Prioritizing Samples in Reinforcement Learning with Reducible Loss

**Authors:** Shivakanth Sujit, Somjit Nath, Pedro Braga, Samira Ebrahimi Kahou

### Privacy Amplification via Compression: Achieving the Optimal Privacy-Accuracy-Communication Trade-off in Distributed Mean Estimation

**Authors:** Wei-Ning Chen, Dan Song, Ayfer Ozgur, Peter Kairouz

### [Spotlight] Private Distribution Learning with Public Data: The View from Sample Compression

**Authors:** Shai Ben-David, Alex Bie, Clément L Canonne, Gautam Kamath, Vikrant Singhal

### Private Federated Frequency Estimation: Adapting to the Hardness of the Instance

**Authors:** Jingfeng Wu, Wennan Zhu, Peter Kairouz, Vladimir Braverman

### Projection-Free Online Convex Optimization via Efficient Newton Iterations

**Authors:** Khashayar Gatmiry, Zak Mhammedi

### ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

**Authors:** Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan LI, Hang Su, Jun Zhu

### Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

**Authors:** Shuhuai Ren, Aston Zhang, Yi Zhu, Shuai Zhang, Shuai Zheng, Mu Li, Alexander Smola, Xu Sun

### PromptRestorer: A Prompting Image Restoration Method with Degradation Perception

**Authors:** Cong Wang, Jinshan Pan, Wei Wang, Jiangxin Dong, Mengzhu Wang, Yakun Ju, Junyang Chen, Xiao-Ming Wu

### ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion

**Authors:** Yingjun Du, Zehao Xiao, Shengcai Liao, Cees Snoek

### Prototypical Variational Autoencoder for 3D Few-shot Object Detection

**Authors:** Weiliang Tang, Biqi YANG, Xianzhi Li, Yun-Hui Liu, Pheng-Ann Heng, Chi-Wing Fu

### Provable Guarantees for Generative Behavior Cloning: Bridging Low-Level Stability and High-Level Behavior

**Authors:** Adam Block, Ali Jadbabaie, Daniel Pfrommer, Max Simchowitz, Russ Tedrake

### Provable convergence guarantees for black-box variational inference

**Authors:** Justin Domke, Robert Gower, Guillaume Garrigos

### Provably Efficient Offline Goal-Conditioned Reinforcement Learning with General Function Approximation and Single-Policy Concentrability

**Authors:** Hanlin Zhu, Amy Zhang

### Provably Fast Convergence of Independent Natural Policy Gradient for Markov Potential Games

**Authors:** Youbang Sun, Tao Liu, Ruida Zhou, P. R. Kumar, Shahin Shahrampour

### Q-DM: An Efficient Low-bit Quantized Diffusion Model

**Authors:** Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, Baochang Zhang

### QH9: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules

**Authors:** Haiyang Yu, Meng Liu, Youzhi Luo, Alex Strasser, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji

### RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths

**Authors:** Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, Ping Luo

### RETVec: Resilient and Efficient Text Vectorizer

**Authors:** Elie Bursztein, Marina Zhang, Owen Vallis, XINYU JIA, Alexey Kurakin

### RGMIL: Guide Your Multiple-Instance Learning Model with Regressor

**Authors:** Zhaolong Du, Shasha Mao, Yimeng Zhang, Shuiping Gou, Licheng Jiao, Lin Xiong

### RH-BrainFS: Regional Heterogeneous Multimodal Brain Networks Fusion Strategy

**Authors:** Hongting Ye, Yalu Zheng, Yueying Li, Ke Zhang, Youyong Kong, Yonggui Yuan

### RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion

**Authors:** Zhuoqun Huang, Neil G Marchant, Keane Lucas, Lujo Bauer, Olga Ohrimenko, Benjamin Rubinstein

### Rank-1 Matrix Completion with Gradient Descent and Small Random Initialization

**Authors:** Daesung Kim, Hye Won Chung

### Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors

**Authors:** Paul Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, ethan cohen, Aidan Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth Norman, Tanishq Abraham

### Recurrent Hypernetworks are Surprisingly Strong in Meta-RL

**Authors:** Jacob Beck, Risto Vuorio, Zheng Xiong, Shimon Whiteson

### Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability

**Authors:** Jishnu Ray Chowdhury, Cornelia Caragea

### Reduced Policy Optimization for Continuous Control with Hard Constraints

**Authors:** Shutong Ding, Jingya Wang, Yali Du, Ye Shi

### Reducing Blackwell and Average Optimality to Discounted MDPs via the Blackwell Discount Factor

**Authors:** Julien Grand-Clément, Marek Petrik

### Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method

**Authors:** Qihang Fang, Yafei Song, Keqiang Li, Liefeng Bo

### RegBN: Batch Normalization of Multimodal Data with Regularization

**Authors:** Morteza Ghahremani Boozandani, Christian Wachinger

### Regression with Cost-based Rejection

**Authors:** Xin Cheng, Yuzhou Cao, Haobo Wang, Hongxin Wei, Bo An, Lei Feng

### [Spotlight] Regret Matching+: (In)Stability and Fast Convergence in Games

**Authors:** Gabriele Farina, Julien Grand-Clément, Christian Kroer, Chung-Wei Lee, Haipeng Luo

### Regret-Optimal Model-Free Reinforcement Learning for Discounted MDPs with Short Burn-In Time

**Authors:** Xiang Ji, Gen Li

### Reinforcement Learning with Simple Sequence Priors

**Authors:** Tankred Saanum, Noemi Elteto, Peter Dayan, Marcel Binz, Eric Schulz

### Reining Generalization in Offline Reinforcement Learning via Representation Distinction

**Authors:** Yi Ma, Hongyao Tang, Dong Li, Zhaopeng Meng

### Relative Entropic Optimal Transport: a (Prior-aware) Matching Perspective to (Unbalanced) Classification

**Authors:** Liangliang Shi, Haoyu Zhen, Gu Zhang, Junchi Yan

### Representation Learning via Consistent Assignment of Views over Random Partitions

**Authors:** Thalles Santos Silva, Adín Ramírez Rivera

### Resilient Constrained Learning

**Authors:** Ignacio Hounie, Alejandro Ribeiro, Luiz F. O. Chamon

### Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis

**Authors:** Victor Letzelter, Mathieu Fontaine, Mickael Chen, Patrick Pérez, Slim Essid, Gaël Richard

### Resolving the Tug-of-War: A Separation of Communication and Learning in Federated Learning

**Authors:** Junyi Li, Heng Huang

### Response Length Perception and  Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline

**Authors:** Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, Yang You

### [Oral] Rethinking Bias Mitigation: Fairer Architectures Make for Fairer Face Recognition

**Authors:** Samuel Dooley, Rhea Sukthanker, John Dickerson, Colin White, Frank Hutter, Micah Goldblum

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5B

### Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective

**Authors:** Chenyu You, Weicheng Dai, Yifei Min, Fenglin Liu, David Clifton, S. Kevin Zhou, Lawrence Staib, James Duncan

### Rethinking Tokenizer and Decoder in Masked Graph Modeling for Molecules

**Authors:** ZHIYUAN LIU, Yaorui Shi, An Zhang, Enzhi Zhang, Kenji Kawaguchi, Xiang Wang, Tat-Seng Chua

### Revisiting Visual Model Robustness: A Frequency Long-Tailed Distribution View

**Authors:** Zhiyu Lin, Yifei Gao, Yunfan Yang, Jitao Sang

### Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery

**Authors:** Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Weinberger

### Riemannian stochastic optimization methods avoid strict saddle points

**Authors:** Ya-Ping Hsieh, Mohammad Reza Karimi Jaghargh, Andreas Krause, Panayotis Mertikopoulos

### Robust Matrix Sensing in the Semi-Random Model

**Authors:** Xing Gao, Yu Cheng

### Robust Mean Estimation Without Moments for Symmetric Distributions

**Authors:** Gleb Novikov, David Steurer, Stefan Tiegel

### Rubik's Cube: High-Order Channel Interactions with a Hierarchical Receptive Field

**Authors:** Naishan Zheng, man zhou, Chong Zhou, Chen Change Loy

### SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models

**Authors:** Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhi-Ming Ma

### SAME: Uncovering GNN Black Box with Structure-aware Shapley-based Multipiece Explanations

**Authors:** Ziyuan Ye, Rihan Huang, Qilin Wu, Quanying Liu

### SOL: Sampling-based Optimal Linear bounding of arbitrary scalar functions

**Authors:** Yuriy Biktairov, Jyotirmoy Deshmukh

### SPA: A Graph Spectral Alignment Perspective for Domain Adaptation

**Authors:** Zhiqing Xiao, Haobo Wang, Ying Jin, Lei Feng, Gang Chen, Fei Huang, Junbo Zhao

### SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning

**Authors:** Dohyeok Lee, Seungyub Han, Taehyun Cho, Jungwoo Lee

### SPRING: Studying Papers and Reasoning to play Games

**Authors:** Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Russ Salakhutdinov, Amos Azaria, Tom Mitchell, Yuanzhi Li

### SQ Lower Bounds for Learning Mixtures of Linear Classifiers

**Authors:** Ilias Diakonikolas, Daniel Kane, Yuxin Sun

### [Spotlight] Saddle-to-Saddle Dynamics in Diagonal Linear Networks

**Authors:** Scott Pesme, Nicolas Flammarion

### Sample Complexity for Quadratic Bandits: Hessian Dependent Bounds and Optimal Algorithms

**Authors:** Qian Yu, Yining Wang, Baihe Huang, Qi Lei, Jason Lee

### Sample based Explanations via Generalized Representers

**Authors:** Che-Ping Tsai, Chih-Kuan Yeh, Pradeep Ravikumar

### Sample-Conditioned Hypothesis Stability Sharpens Information-Theoretic Generalization Bounds

**Authors:** Ziqiao Wang, Yongyi Mao

### [Oral] Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent

**Authors:** Jihao Andreas Lin, Javier Antorán, Shreyas Padhy, David Janz, José Miguel Hernández-Lobato, Alexander Terenin

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5C

### Scaling Laws for Hyperparameter Optimization

**Authors:** Arlind Kadra, Maciej Janowski, Martin Wistuba, Josif Grabocka

### [Spotlight] Scaling Open-Vocabulary Object Detection

**Authors:** Matthias Minderer, Alexey Gritsenko, Neil Houlsby

### Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer

**Authors:** Yuandong Tian, Yiping Wang, Beidi Chen, Simon Du

### Scattering Vision Transformer: Spectral Mixing Matters

**Authors:** Badri Patro, Vijay Agneeswaran

### [Spotlight] Segment Any Point Cloud Sequences by Distilling Vision Foundation Models

**Authors:** Youquan Liu, Lingdong Kong, Jun CEN, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu

### [Spotlight] Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models

**Authors:** Alvin Heng, Harold Soh

### Self-Predictive Universal AI

**Authors:** Elliot Catt, Jordi Grau-Moya, Marcus Hutter, Matthew Aitchison, Tim Genewein, Grégoire Delétang, Kevin Li, Joel Veness

### Self-Supervised Motion Magnification by Backpropagating Through Optical Flow

**Authors:** Zhaoying Pan, Daniel Geng, Andrew Owens

### Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation

**Authors:** Weihang Dai, Yao DU, Hanru Bai, Kwang-Ting Cheng, Xiaomeng Li

### Sensitivity in Translation Averaging

**Authors:** Lalit Manam, Venu Madhav Govindu

### Shape Non-rigid Kinematics (SNK): A Zero-Shot Method for Non-Rigid Shape Matching via Unsupervised Functional Map Regularized Reconstruction

**Authors:** Souhaib Attaiki, Maks Ovsjanikov

### Sharp Bounds for Generalized Causal Sensitivity Analysis

**Authors:** Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel

### ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer

**Authors:** Haoran You, Huihong Shi, Yipin Guo, Celine Lin

### [Spotlight] SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling

**Authors:** Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long

### Similarity-based cooperative equilibrium

**Authors:** Caspar Oesterheld, Johannes Treutlein, Roger Grosse, Vincent Conitzer, Jakob Foerster

### Simple and Controllable Music Generation

**Authors:** Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, Gabriel Synnaeve, Yossi Adi, Alexandre Defossez

### Simplifying Neural Network Training Under Class Imbalance

**Authors:** Ravid Shwartz-Ziv, Micah Goldblum, Yucen Li, C. Bayan Bruss, Andrew Wilson

### Simultaneous embedding of multiple attractor manifolds in a recurrent neural network using constrained gradient optimization

**Authors:** Haggai Agmon, Yoram Burak

### Single-Pass Pivot Algorithm for Correlation Clustering. Keep it simple!

**Authors:** Konstantin Makarychev, Sayak Chakrabarty

### Sketching Algorithms for Sparse Dictionary Learning: PTAS and Turnstile Streaming

**Authors:** Gregory Dexter, Petros Drineas, David Woodruff, Taisuke Yasuda

### Slot-guided Volumetric Object Radiance Fields

**Authors:** DI QI, Tong Yang, Xiangyu Zhang

### Social Motion Prediction with Cognitive Hierarchies

**Authors:** Wentao Zhu, Jason Qin, Yuke Lou, Hang Ye, Xiaoxuan Ma, Hai Ci, Yizhou Wang

### Soft-Unification in Deep Probabilistic Logic

**Authors:** Jaron Maene, Luc De Raedt

### Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks

**Authors:** Changhyeon Lee, Seulki Lee

### SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data

**Authors:** BANG AN, Xun Zhou, YONGJIAN ZHONG, Tianbao Yang

### Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning

**Authors:** Ronald Xie, Kuan Pang, Sai Chung, Catia Perciani, Sonya MacParland, Bo Wang, Gary Bader

### Spectral Evolution and Invariance in Linear-width Neural Networks

**Authors:** Zhichao Wang, Andrew Engel, Anand D Sarwate, Ioana Dumitriu, Tony Chiang

### Spontaneous symmetry breaking in generative diffusion models

**Authors:** Gabriel Raya, Luca Ambrogioni

### Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases

**Authors:** Mazda Moayeri, Wenxiao Wang, Sahil Singla, Soheil Feizi

### Stability and Generalization of the Decentralized Stochastic Gradient Descent Ascent Algorithm

**Authors:** Miaoxi Zhu, Li Shen, Bo Du, Dacheng Tao

### State-Action Similarity-Based Representations for Off-Policy Evaluation

**Authors:** Brahma Pavse, Josiah Hanna

### State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory

**Authors:** Shida Wang, Beichen Xue

### Statistical Insights into HSIC in High Dimensions

**Authors:** Tao Zhang, Yaowu Zhang, Tingyou Zhou

### Statistical Knowledge Assessment for Large Language Models

**Authors:** Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, Lei Li

### Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks

**Authors:** Feng Chen, Daniel Kunin, Atsushi Yamamura, Surya Ganguli

### Strategyproof Voting under Correlated Beliefs

**Authors:** Daniel Halpern, Rachel Li, Ariel Procaccia

### Streaming Algorithms and Lower Bounds for Estimating Correlation Clustering Cost

**Authors:** Sepehr Assadi, Vihan Shah, Chen Wang

### Structure of universal formulas

**Authors:** Dmitry Yarotsky

### [Oral] Students Parrot Their Teachers: Membership Inference on Model Distillation

**Authors:** Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A. Choquette-Choo, Nicholas Carlini, Florian Tramer

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5B

### StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

**Authors:** Yinghao Aaron Li, Cong Han, Vinay Raghavan, Gavin Mischler, Nima Mesgarani

### Successor-Predecessor Intrinsic Exploration

**Authors:** Changmin Yu, Neil Burgess, Maneesh Sahani, Samuel J Gershman

### Supported Value Regularization for Offline Reinforcement Learning

**Authors:** Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, Xiangyang Ji

### SutraNets: Sub-series Autoregressive Networks for Long-Sequence, Probabilistic Forecasting

**Authors:** Shane Bergsma, Tim Zeyl, Lei Guo

### Swarm Reinforcement Learning for Adaptive Mesh Refinement

**Authors:** Niklas Freymuth, Philipp Dahlinger, Tobias Würth, Simon Reisch, Luise Kärger, Gerhard Neumann

### Switching Autoregressive Low-rank Tensor Models

**Authors:** Hyun Dong Lee, andrew warrington, Joshua Glaser, Scott Linderman

### Symbolic Discovery of Optimization Algorithms

**Authors:** Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V Le

### Synthetic Combinations: A Causal Inference Framework for Combinatorial Interventions

**Authors:** Abhineet Agarwal, Anish Agarwal, Suhas Vijaykumar

### TWIGMA: A dataset of AI-Generated Images with Metadata From Twitter

**Authors:** Yiqun Chen, James Zou

### TabMT: Generating tabular data with masked transformers

**Authors:** Manbir Gulati, Paul Roysdon

### Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation: Minimax Optimal and Instance-Dependent Regret Bounds

**Authors:** Jiayi Huang, Han Zhong, Liwei Wang, Lin Yang

### Taylor TD-learning

**Authors:** Michele Garibbo, Maxime Robeyns, Laurence Aitchison

### Team-PSRO for Learning Approximate TMECor in Large Team Games via Cooperative Reinforcement Learning

**Authors:** Stephen McAleer, Gabriele Farina, Gaoyue Zhou, Mingzhi Wang, Yaodong Yang, Tuomas Sandholm

### [Spotlight] Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training

**Authors:** Yefan Zhou, TIANYU PANG, Keqin Liu, charles martin, Michael Mahoney, Yaoqing Yang

### Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification

**Authors:** Kanishk Jain, Shyamgopal Karthik, Vineet Gandhi

### Test-time Adaptation of Discriminative Models via Diffusion Generative Feedback

**Authors:** Mihir Prabhudesai, Tsung-Wei Ke, Alex Li, Deepak Pathak, Katerina Fragkiadaki

### Text-to-Image Diffusion Models are Zero Shot Classifiers

**Authors:** Kevin Clark, Priyank Jaini

### The CLIP Model is Secretly an Image-to-Prompt Converter

**Authors:** Yuxuan Ding, Chunna Tian, Haoxuan Ding, Lingqiao Liu

### The Learnability of In-Context Learning

**Authors:** Noam Wies, Yoav Levine, Amnon Shashua

### The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification

**Authors:** Linhao Qu, xiaoyuan luo, Kexue Fu, Manning Wang, Zhijian Song

### [Spotlight] Theoretical and Practical Perspectives on what Influence Functions Do

**Authors:** Andrea Schioppa, Katja Filippova, Ivan Titov, Polina Zablotskaia

### Theoretically Guaranteed Bidirectional Data Rectification for Robust Sequential Recommendation

**Authors:** Yatong Sun, Bin Wang, Zhu Sun, Xiaochun Yang, Yan Wang

### Thinker: Learning to Plan and Act

**Authors:** Stephen Chung, Ivan Anokhin, David Krueger

### Toward Re-Identifying Any Animal

**Authors:** Bingliang Jiao, Lingqiao Liu, Liying Gao, Ruiqi Wu, Guosheng Lin, PENG WANG, Yanning Zhang

### Towards Optimal Caching and Model Selection for Large Model Inference

**Authors:** Banghua Zhu, Ying Sheng, Lianmin Zheng, Clark Barrett, Michael Jordan, Jiantao Jiao

### Towards robust and generalizable representations of extracellular data using contrastive learning

**Authors:** Ankit Vishnubhotla, Charlotte Loh, Akash Srivastava, Liam Paninski, Cole Hurwitz

### Towards the Difficulty for a Deep Neural Network to Learn Concepts of Different Complexities

**Authors:** Dongrui Liu, Huiqi Deng, Xu Cheng, Qihan Ren, Kangrui Wang, Quanshi Zhang

### Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks

**Authors:** Jun Yin, Chaozhuo Li, Hao Yan, Jianxun Lian, Senzhang Wang

### Training Transitive and Commutative Multimodal Transformers with LoReTTa

**Authors:** Manuel Tran, Yashin Dicente Cid, Amal Lahiani, Fabian Theis, Tingying Peng, Eldad Klaiman

### Transformed Low-Rank Parameterization Can Help Robust Generalization for Tensor Neural Networks

**Authors:** Andong Wang, Chao Li, Mingyuan Bai, Zhong Jin, Guoxu Zhou, Qibin Zhao

### Transformers over Directed Acyclic Graphs

**Authors:** Yuankai Luo, Veronika Thost, Lei Shi

### [Spotlight] Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction

**Authors:** Anagh Malik, Parsa Mirdehghan, Sotiris Nousias, Kyros Kutulakos, David Lindell

### TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion

**Authors:** Preetha Vijayan, Prashant Bhat, Bahram Zonooz, Elahe Arani

### Triple Eagle: Simple, Fast and Practical Budget-Feasible Mechanisms

**Authors:** Kai Han, You Wu, He Huang, Shuang Cui

### Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection

**Authors:** Hezhe Qiao, Guansong Pang

### Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints

**Authors:** Dohyeong Kim, Kyungjae Lee, Songhwai Oh

### Two Sides of One Coin: the Limits of Untuned SGD and the Power of Adaptive Methods

**Authors:** Junchi YANG, Xiang Li, Ilyas Fatkhullin, Niao He

### UE4-NeRF:Neural Radiance Field for Real-Time Rendering of Large-Scale Scene

**Authors:** Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, Mohammed Bennamoun

### Unbiased constrained sampling with Self-Concordant Barrier Hamiltonian Monte Carlo

**Authors:** Maxence Noble, Valentin De Bortoli, Alain Durmus

### Uncertainty Quantification via Neural Posterior Principal Components

**Authors:** Elias Nehme, Omer Yair, Tomer Michaeli

### Uncertainty-Aware Alignment  Network  for Cross-Domain Video-Text Retrieval

**Authors:** Xiaoshuai Hao, Wanqian Zhang

### Unconstrained Dynamic Regret via Sparse Coding

**Authors:** Zhiyu Zhang, Ashok Cutkosky, Yannis Paschalidis

### [Spotlight] Understanding Multi-phase Optimization Dynamics and Rich Nonlinear Behaviors of ReLU Networks

**Authors:** Mingze Wang, Chao Ma

### Understanding and Improving Feature Learning for Out-of-Distribution Generalization

**Authors:** Yongqiang Chen, Wei Huang, Kaiwen Zhou, Yatao Bian, Bo Han, James Cheng

### UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition

**Authors:** qiufu li, Xi Jia, Jiancan Zhou, Linlin Shen, Jinming Duan

### [Spotlight] Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems

**Authors:** Benjamin Coleman, Wang-Cheng Kang, Matthew Fahrbach, Ruoxi Wang, Lichan Hong, Ed Chi, Derek Cheng

### Unified Enhancement of Privacy Bounds for Mixture Mechanisms via $f$-Differential Privacy

**Authors:** Chendi Wang, Buxin Su, Jiayuan Ye, Reza Shokri, Weijie Su

### Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective

**Authors:** Zeyu Zhang, Yi Su, Hui Yuan, Yiran Wu, Rishab Balasubramanian, Qingyun Wu, Huazheng Wang, Mengdi Wang

### Unleash the Potential of Image Branch for Cross-modal 3D Object Detection

**Authors:** Yifan Zhang, Qijian Zhang, Junhui Hou, Yixuan Yuan, Guoliang Xing

### Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift

**Authors:** Yongduo Sui, Qitian Wu, Jiancan Wu, Qing Cui, Longfei Li, Jun Zhou, Xiang Wang, Xiangnan He

### Unleashing the Power of Randomization in Auditing Differentially Private ML

**Authors:** Krishna Pillutla, Galen Andrew, Peter Kairouz, H. Brendan McMahan, Alina Oprea, Sewoong Oh

### Unlimiformer: Long-Range Transformers with Unlimited Length Input

**Authors:** Amanda Bertsch, Uri Alon, Graham Neubig, Matthew Gormley

### Unsupervised Learning for Solving the Travelling Salesman Problem

**Authors:** Yimeng Min, Yiwei Bai, Carla Gomes

### Use perturbations when learning from explanations

**Authors:** Juyeon Heo, Vihari Piratla, Matthew Wicker, Adrian Weller

### Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models

**Authors:** Naoki Egami, Musashi Hinck, Brandon Stewart, Hanying Wei

### VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models

**Authors:** Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

### VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning

**Authors:** Jiayi Guan, Guang Chen, Jiaming Ji, Long Yang, ao zhou, Zhijun Li, changjun jiang

### [Spotlight] VaRT: Variational Regression Trees

**Authors:** Sebastian Salazar

### Variational Gaussian Processes with Decoupled Conditionals

**Authors:** Xinran Zhu, Kaiwen Wu, Natalie Maus, Jacob Gardner, David Bindel

### Versatile Energy-Based Probabilistic Models for High Energy Physics

**Authors:** Taoli Cheng, Aaron Courville

### Video Timeline Modeling For News Story Understanding

**Authors:** Meng Liu, Mingda Zhang, Jialu Liu, Hanjun Dai, Ming-Hsuan Yang, Shuiwang Ji, Zheyun Feng, Boqing Gong

### VideoComposer: Compositional Video Synthesis with Motion Controllability

**Authors:** Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou

### Visual Instruction Inversion: Image Editing via Image Prompting

**Authors:** Thao Nguyen, Yuheng Li, Utkarsh Ojha, Yong Jae Lee

### [Oral] Visual Instruction Tuning

**Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5D

### Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

**Authors:** Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, Wei-Ning Hsu

### Wasserstein Gradient Flows for Optimizing Gaussian Mixture Policies

**Authors:** Hanna Ziesche, Leonel Rozo

### [Spotlight] Wasserstein Quantum Monte Carlo: A Novel Approach for Solving the Quantum Many-Body Schrödinger Equation

**Authors:** Kirill Neklyudov, Jannes Nys, Luca Thiede, Juan Carrasquilla, Qiang Liu, Max Welling, Alireza Makhzani

### What Makes Good Examples for Visual In-Context Learning?

**Authors:** Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu

### When Do Graph Neural Networks Help with Node Classification? Investigating the Homophily Principle on Node Distinguishability

**Authors:** Sitao Luan, Chenqing Hua, Minkai Xu, Qincheng Lu, Jiaqi Zhu, Xiao-Wen Chang, Jie Fu, Jure Leskovec, Doina Precup

### When are ensembles really effective?

**Authors:** Ryan Theisen, Hyunsuk Kim, Yaoqing Yang, Liam Hodgkinson, Michael Mahoney

### Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations

**Authors:** Piotr Indyk, Haike Xu

### xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data

**Authors:** Jing Gong, Minsheng Hao, Xingyi Cheng, Xin Zeng, Chiming Liu, Jianzhu Ma, Xuegong Zhang, Taifeng Wang, Le Song

</details>

<details><summary><h3 style='display: inline;'> Poster Session 6: Thursday, Dec 14, 15:00 CT</h3></summary>

### $S^3$: Increasing GPU Utilization during Generative Inference for Higher Throughput

**Authors:** Yunho Jin, Chun-Feng Wu, David Brooks, Gu-Yeon Wei

### $\mathbf{\mathbb{E}^{FWI}}$: Multiparameter Benchmark Datasets for Elastic Full Waveform Inversion of Geophysical Properties

**Authors:** Shihang Feng, Hanchen Wang, Chengyuan Deng, Yinan Feng, Yanhua Liu, Min Zhu, Peng Jin, Yinpeng Chen, Youzuo Lin

### $\mathcal{M}^4$: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models

**Authors:** Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju, Haoyi Xiong

### (Amplified) Banded Matrix Factorization: A unified approach to private training

**Authors:** Christopher A. Choquette-Choo, Arun Ganesh, Ryan McKenna, H. Brendan McMahan, John Rush, Abhradeep Guha Thakurta, Zheng Xu

### A Bounded Ability Estimation for Computerized Adaptive Testing

**Authors:** Yan Zhuang, Qi Liu, Guanhao Zhao, Zhenya Huang, Weizhe Huang, Zachary Pardos, Enhong Chen, Jinze Wu, Xin Li

### A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP)

**Authors:** Weijie Tu, Weijian Deng, Tom Gedeon

### A Comprehensive Benchmark for Neural Human Radiance Fields

**Authors:** Kenkun Liu, Derong Jin, Ailing Zeng, Xiaoguang Han, Lei Zhang

### A Dataset for Analyzing Streaming Media Performance over HTTP/3 Browsers

**Authors:** Sapna Chaudhary, Mukulika Maity, Sandip Chakraborty, Naval Shukla

### A Dataset of Relighted 3D Interacting Hands

**Authors:** Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe De Bree, Tomas Simon, Bo Peng, Shubham Garg, Kevyn McPhail, Takaaki Shiratori

### A Logic for Expressing Log-Precision Transformers

**Authors:** William Merrill, Ashish Sabharwal

### A Massive Scale Semantic Similarity Dataset of Historical English

**Authors:** Emily Silcock, Abhishek Arora, Melissa Dell

### A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks

**Authors:** Vignesh Kothapalli, Tom Tirer, Joan Bruna

### A Novel Approach for Effective Multi-View Clustering with Information-Theoretic Perspective

**Authors:** Chenhang Cui, Yazhou Ren, Jingyu Pu, Jiawei Li, Xiaorong Pu, Tianyi Wu, Yutao Shi, Lifang He

### A Novel Framework for Policy Mirror Descent with General Parameterization and Linear Convergence

**Authors:** Carlo Alfano, Rui Yuan, Patrick Rebeschini

### A Regularized Conditional GAN for Posterior Sampling in Image Recovery Problems

**Authors:** Matthew Bendel, Rizwan Ahmad, Philip Schniter

### [Spotlight] A Robust and Opponent-Aware League Training Method for StarCraft II

**Authors:** Ruozi Huang, Xipeng Wu, Hongsheng Yu, Zhong Fan, Haobo Fu, Qiang Fu, Wei Yang

### A Step Towards Worldwide Biodiversity Assessment:  The BIOSCAN-1M Insect Dataset

**Authors:** Zahra Gharaee, ZeMing Gong, Nicholas Pellegrino, Iuliia Zarubiieva, Joakim Bruslund Haurum, Scott Lowe, Jaclyn McKeown, Chris Ho, Joschka McLeod, Yi-Yun Wei, Jireh Agda, Sujeevan Ratnasingham, Dirk Steinke, Angel Chang, Graham Taylor, Paul Fieguth

### A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning

**Authors:** Florian Felten, Lucas N. Alegre, Ann Nowe, Ana Bazzan, El Ghazali Talbi, Grégoire Danoy, Bruno da Silva

### A Unified Framework for Uniform Signal Recovery in Nonlinear Generative Compressed Sensing

**Authors:** Junren Chen, Jonathan Scarlett, Michael Ng, Zhaoqiang Liu

### A Unified, Scalable Framework for Neural Population Decoding

**Authors:** Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake Richards, Matthew Perich, Guillaume Lajoie, Eva Dyer

### A benchmark of categorical encoders for binary classification

**Authors:** Federico Matteucci, Vadim Arzamasov, Klemens Böhm

### A generative model of the hippocampal formation trained with theta driven local learning rules

**Authors:** Tom M George, Kimberly Stachenfeld, Caswell Barry, Claudia Clopath, Tomoki Fukai

### ADGym: Design Choices for Deep Anomaly Detection

**Authors:** Minqi Jiang, Chaochuan Hou, Ao Zheng, Songqiao Han, Hailiang Huang, Qingsong Wen, Xiyang Hu, Yue Zhao

### AQuA: A Benchmarking Tool for Label Quality Assessment

**Authors:** Mononito Goswami, Vedant Sanil, Arjun Choudhry, Arvind Srinivasan, Chalisa Udompanyawit, Artur Dubrawski

### AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis

**Authors:** Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu

### AVOIDDS: Aircraft Vision-based Intruder Detection Dataset and Simulator

**Authors:** Elysia Smyers, Sydney Katz, Anthony Corso, Mykel J Kochenderfer

### [Spotlight] Accelerated Quasi-Newton Proximal Extragradient: Faster Rate for Smooth Convex Optimization

**Authors:** Ruichen Jiang, Aryan Mokhtari

### Achieving Cross Modal Generalization with Multimodal Unified Representation

**Authors:** Yan Xia, Hai Huang, Jieming Zhu, Zhou Zhao

### Act As You Wish: Fine-Grained Control of Motion Diffusion Model with Hierarchical Semantic Graphs

**Authors:** Peng Jin, Yang Wu, Yanbo Fan, Zhongqian Sun, Wei Yang, Li Yuan

### Active Learning for Semantic Segmentation with Multi-class Label Query

**Authors:** Sehyun Hwang, Sohyun Lee, Hoyoung Kim, Minhyeon Oh, Jungseul Ok, Suha Kwak

### Adaptive Algorithms for Relaxed Pareto Set Identification

**Authors:** Cyrille KONE, Emilie Kaufmann, Laura Richert

### Adaptive Linear Estimating Equations

**Authors:** Mufang Ying, Koulik Khamaru, Cun-Hui Zhang

### [Spotlight] Adaptive whitening with fast gain modulation and slow synaptic plasticity

**Authors:** Lyndon Duong, Eero Simoncelli, Dmitri Chklovskii, David Lipshutz

### Adversarial Resilience in Sequential Prediction via Abstention

**Authors:** Surbhi Goel, Steve Hanneke, Shay Moran, Abhishek Shetty

### [Spotlight] Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach

**Authors:** Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay

### Alexa Arena: A User-Centric Interactive Platform for Embodied AI

**Authors:** Qiaozi Gao, Govindarajan Thattai, Suhaila Shakiah, Xiaofeng Gao, Shreyas Pansare, Vasu Sharma, Gaurav Sukhatme, Hangjie Shi, Bofei Yang, Desheng Zhang, Lucy Hu, Karthika Arumugam, Shui Hu, Matthew Wen, Dinakar Guthy, Shunan Chung, Rohan Khanna, Osman Ipek, Leslie Ball, Kate Bland, Heather Rocker, Michael Johnston, Reza Ghanadan, Dilek Hakkani-Tur, Prem Natarajan

### Algorithm Selection for Deep Active Learning with Imbalanced Datasets

**Authors:** Jifan Zhang, Shuai Shao, Saurabh Verma, Robert Nowak

### All Points Matter: Entropy-Regularized Distribution Alignment for Weakly-supervised 3D Segmentation

**Authors:** Liyao Tang, Zhe Chen, Shanshan Zhao, Chaoyue Wang, Dacheng Tao

### Alternating Gradient Descent and Mixture-of-Experts for Integrated Multimodal Perception

**Authors:** Hassan Akbari, Dan Kondratyuk, Yin Cui, Rachel Hornung, Huisheng Wang, Hartwig Adam

### Ambient Diffusion: Learning Clean Distributions from Corrupted Data

**Authors:** Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, Adam Klivans

### American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers

**Authors:** Melissa Dell, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora, Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin, Leander Heldring

### An Alternating Optimization Method for Bilevel Problems under the Polyak-Łojasiewicz Condition

**Authors:** Quan Xiao, Songtao Lu, Tianyi Chen

### An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint

**Authors:** Mengzhao Wang, Lingwei Lv, Xiaoliang Xu, Yuxiang Wang, Qiang Yue, Jiongkang Ni

### An Empirical Investigation of the Role of Pre-training in Lifelong Learning

**Authors:** Sanket Vaibhav Mehta, Darshan Patil, Sarath Chandar, Emma Strubell

### An Exploration-by-Optimization Approach to Best of Both Worlds in Linear Bandits

**Authors:** Shinji Ito, Kei Takemura

### An NLP Benchmark Dataset for Assessing Corporate Climate Policy Engagement

**Authors:** Gaku Morio, Christopher D Manning

### An Optimization-based Approach To Node Role Discovery in Networks: Approximating Equitable Partitions

**Authors:** Michael Scholkemper, Michael T Schaub

### AndroidInTheWild: A Large-Scale Dataset For Android Device Control

**Authors:** Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, Timothy Lillicrap

### Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation

**Authors:** Binhui Xie, Shuang Li, Qingju Guo, Chi Liu, Xinjing Cheng

### [Oral] Are Emergent Abilities of Large Language Models a Mirage?

**Authors:** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6A

### Asymptotics of Bayesian Uncertainty Estimation in Random Features Regression

**Authors:** Youngsoo Baek, Samuel Berchuck, Sayan Mukherjee

### Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection

**Authors:** suresh kumar amalapuram, Sumohana Channappayya, Bheemarjuna Reddy Tamma

### Auslan-Daily: Australian Sign Language Translation for Daily Communication and News

**Authors:** Xin Shen, Shaozu Yuan, Hongwei Sheng, Heming Du, Xin Yu

### Automated Classification of Model Errors on ImageNet

**Authors:** Momchil Peychev, Mark Müller, Marc Fischer, Martin Vechev

### Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective

**Authors:** Yifei Wang, Liangchen Li, Jiansheng Yang, Zhouchen Lin, Yisen Wang

### Balanced Training for Sparse GANs

**Authors:** Yite Wang, Jing Wu, NAIRA HOVAKIMYAN, Ruoyu Sun

### BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis

**Authors:** Zelin Ni, Hang Yu, Shizhan Liu, Jianguo Li, Weiyao Lin

### Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks

**Authors:** Micah Goldblum, Hossein Souri, Renkun Ni, Manli Shu, Viraj Prabhu, Gowthami Somepalli, Prithvijit Chattopadhyay, Mark Ibrahim, Adrien Bardes, Judy Hoffman, Rama Chellappa, Andrew Wilson, Tom Goldstein

### BayesDAG: Gradient-Based Posterior Inference for Causal Discovery

**Authors:** Yashas Annadani, Nick Pawlowski, Joel Jennings, Stefan Bauer, Cheng Zhang, Wenbo Gong

### BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset

**Authors:** Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, Yaodong Yang

### Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback

**Authors:** Jangwon Kim, Hangyeol Kim, Jiwook Kang, Jongchan Baek, Soohee Han

### BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing

**Authors:** Subhro Roy, Samuel Thomson, Tongfei Chen, Richard Shin, Adam Pauls, Jason Eisner, Benjamin Van Durme

### Benchmark of Machine Learning Force Fields for Semiconductor Simulations: Datasets, Metrics, and Comparative Analysis

**Authors:** Geonu Kim, Byunggook Na, Gunhee Kim, Hyuntae Cho, Seungjin Kang, Hee Sun Lee, Saerom Choi, Heejae Kim, Seungwon Lee, Yongdeok Kim

### Benchmarking Distribution Shift in Tabular Data with TableShift

**Authors:** Josh Gardner, Zoran Popovic, Ludwig Schmidt

### Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Bone Shape Reconstruction

**Authors:** Mahesh Shakya, Bishesh Khanal

### Benchmarking Foundation Models with Language-Model-as-an-Examiner

**Authors:** Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, Lei Hou

### Benchmarking Large Language Models on CMExam - A comprehensive Chinese Medical Exam Dataset

**Authors:** Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, LEI ZHU, Michael Lingzhi Li

### Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models

**Authors:** Shuo Chen, Jindong Gu, Zhen Han, Yunpu Ma, Philip Torr, Volker Tresp

### Benchmarking Robustness to Adversarial Image Obfuscations

**Authors:** Florian Stimberg, Ayan Chakrabarti, Chun-Ta Lu, Hussein Hazimeh, Otilia Stretcu, Wei Qiao, Yintao Liu, Merve Kaya, Cyrus Rashtchian, Ariel Fuxman, Mehmet Tek, Sven Gowal

### Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase

**Authors:** Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Sida Peng, Yujun Shen

### Beyond MLE: Convex Learning for Text Generation

**Authors:** Chenze Shao, Zhengrui Ma, Min Zhang, Yang Feng

### Beyond Unimodal: Generalising Neural Processes for Multimodal Uncertainty Estimation

**Authors:** Myong Chol Jung, He Zhao, Joanna Dipnall, Lan Du

### Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start

**Authors:** Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

### BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series

**Authors:** Andrea Nascetti, Ritu Yadav, Kirill Brodt, Qixun Qu, Hongwei Fan, Yuri Shendryk, Isha Shah, Christine Chung

### Bitstream-Corrupted Video Recovery: A Novel Benchmark Dataset and Method

**Authors:** Tianyi Liu, Kejun Wu, Yi Wang, Wenyang Liu, Kim-Hui Yap, Lap-Pui Chau

### Boosting Spectral Clustering on Incomplete Data via Kernel Correction and Affinity Learning

**Authors:** Fangchen Yu, Runze Zhao, Zhan Shi, Yiwen Lu, Jicong Fan, Yicheng Zeng, Jianfeng Mao, Wenye Li

### Breadcrumbs to the Goal: Supervised Goal Selection from Human-in-the-Loop Feedback

**Authors:** Marcel Torne Villasevil, Max Balsells I Pamies, Zihan Wang, Samedh Desai, Tao Chen, Pulkit Agrawal, Abhishek Gupta

### [Oral] Bridging RL Theory and Practice with the Effective Horizon

**Authors:** Cassidy Laidlaw, Stuart J Russell, Anca Dragan

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6B

### Building Socio-culturally Inclusive Stereotype Resources with Community Engagement

**Authors:** Sunipa Dev, Jaya Goyal, Dinesh Tewari, Shachi Dave, Vinodkumar Prabhakaran

### Building the Bridge of Schrödinger: A Continuous Entropic Optimal Transport Benchmark

**Authors:** Nikita Gushchin, Alexander Kolesov, Petr Mokrov, Polina Karpikova, Andrei Spiridonov, Evgeny Burnaev, Alexander Korotin

### BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting

**Authors:** Patrick Emami, Abhijeet Sahu, Peter Graf

### Bullying10K: A Large-Scale Neuromorphic Dataset towards Privacy-Preserving Bullying Recognition

**Authors:** Yiting Dong, Yang Li, Dongcheng Zhao, Guobin Shen, Yi Zeng

### [Spotlight] Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes

**Authors:** Yizi Zhang, Tianxiao He, Julien Boussard, Charles Windolf, Olivier Winter, Eric Trautmann, Noam Roth, Hailey Barrell, Mark Churchland, Nicholas A Steinmetz, Erdem Varol, Cole Hurwitz, Liam Paninski

### CAPP-130: A Corpus of Chinese Application Privacy Policy Summarization and Interpretation

**Authors:** pengyun zhu, Long Wen, Jinfei Liu, Feng Xue, Jian Lou, Zhibo Wang, Kui Ren

### CARE-MI: Chinese Benchmark for Misinformation Evaluation in Maternity and Infant Care

**Authors:** Tong Xiang, Liangzhi Li, Wangyue Li, Mingbai Bai, Lu Wei, Bowen Wang, Noa Garcia

### CEIL: Generalized Contextual Imitation Learning

**Authors:** Jinxin Liu, Li He, Yachen Kang, Zifeng Zhuang, Donglin Wang, Huazhe Xu

### CHAMMI: A benchmark for channel-adaptive models in microscopy imaging

**Authors:** Zitong Sam Chen, Chau Pham, Siqi Wang, Michael Doron, Nikita Moshkov, Bryan Plummer, Juan C. Caicedo

### CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation

**Authors:** Xiuzhe Wu, Peng Dai, Weipeng DENG, Handi Chen, Yang Wu, Yan-Pei Cao, Ying Shan, Xiaojuan Qi

### CMMA: Benchmarking Multi-Affection Detection in Chinese Multi-Modal Conversations

**Authors:** Yazhou Zhang, Yang Yu, Qing Guo, Benyou Wang, Dongming Zhao, Sagar Uprety, Dawei Song, Qiuchi Li, Jing Qin

### COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs

**Authors:** Tiep Le, VASUDEV LAL, Phillip Howard

### CQM: Curriculum Reinforcement Learning with a Quantized World Model

**Authors:** Seungjae Lee, Daesol Cho, Jonghae Park, H. Jin Kim

### CSMeD: Bridging the Dataset Gap in Automated Citation Screening for Systematic Literature Reviews

**Authors:** Wojciech Kusa, Oscar E. Mendoza, Matthias Samwald, Petr Knoth, Allan Hanbury

### CaMP: Causal Multi-policy Planning for Interactive Navigation in  Multi-room Scenes

**Authors:** Xiaohan Wang, Yuehu Liu, Xinhang Song, Beibei Wang, Shuqiang Jiang

### Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability

**Authors:** Maciej Falkiewicz, Naoya Takeishi, Imahn Shekhzadeh, Antoine Wehenkel, Arnaud Delaunoy, Gilles Louppe, Alexandros Kalousis

### Calibrating “Cheap Signals” in Peer Review without a Prior

**Authors:** Yuxuan Lu, Yuqing Kong

### Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs

**Authors:** Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Ma Chenhao, Guoliang Li, Kevin Chang, Fei Huang, Reynold Cheng, Yongbin Li

### Can You Rely on Your Model Evaluation? Improving Model Evaluation with Synthetic Test Data

**Authors:** Boris van Breugel, Nabeel Seedat, Fergus Imrie, Mihaela van der Schaar

### Canonical normalizing flows for manifold learning

**Authors:** Kyriakos Flouris, Ender Konukoglu

### Characterization and Learning of Causal Graphs with Small Conditioning Sets

**Authors:** Murat Kocaoglu

### ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling

**Authors:** Tung Nguyen, Jason Jewik, Hritik Bansal, Prakhar Sharma, Aditya Grover

### Closing the gap between the upper bound and lower bound of Adam's iteration complexity

**Authors:** Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, Wei Chen

### Cocktail: Mixing Multi-Modality Control for Text-Conditional Image Generation

**Authors:** Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, Tat-Jen Cham

### Collaboratively Learning Linear Models with Structured Missing Data

**Authors:** Chen Cheng, Gary Cheng, John Duchi

### Color Equivariant Convolutional Networks

**Authors:** Attila Lengyel, Ombretta Strafforello, Robert-Jan Bruintjes, Alexander Gielisse, Jan van Gemert

### Combinatorial Optimization with Policy Adaptation using Latent Space Search

**Authors:** Felix Chalumeau, Shikha Surana, Clément Bonnet, Nathan Grinsztajn, Arnu Pretorius, Alexandre Laterre, Tom Barrett

### Compact Neural Volumetric Video Representations with Dynamic Codebooks

**Authors:** Haoyu Guo, Sida Peng, Yunzhi Yan, Linzhan Mou, Yujun Shen, Hujun Bao, Xiaowei Zhou

### Comparing Apples to Oranges: Learning Similarity Functions for Data Produced by Different Distributions

**Authors:** Leonidas Tsepenekas, Ivan Brugere, Freddy Lecue, Daniele Magazzeni

### Complementary Benefits of Contrastive Learning and Self-Training Under Distribution Shift

**Authors:** Saurabh Garg, Amrith Setlur, Zachary Lipton, Sivaraman Balakrishnan, Virginia Smith, Aditi Raghunathan

### Complexity of Derivative-Free Policy Optimization for Structured $\mathcal{H}_\infty$ Control

**Authors:** Xingang Guo, Darioush Keivan, Geir Dullerud, Peter Seiler, Bin Hu

### Compositional Sculpting of Iterative Generative Processes

**Authors:** Timur Garipov, Sebastiaan De Peuter, Ge Yang, Vikas Garg, Samuel Kaski, Tommi Jaakkola

### Computing Optimal Equilibria and Mechanisms via Learning in Zero-Sum Extensive-Form Games

**Authors:** Brian Zhang, Gabriele Farina, Ioannis Anagnostides, Federico Cacciamani, Stephen McAleer, Andreas Haupt, Andrea Celli, Nicola Gatti, Vincent Conitzer, Tuomas Sandholm

### Concentration analysis of multivariate elliptic diffusions

**Authors:** Lukas Trottner, Cathrine Aeckerle-Willems, Claudia Strauch

### Conditional Distribution Function Estimation Using Neural Networks for Censored and Uncensored Data

**Authors:** Bingqing Hu, Bin Nan

### Conditional score-based diffusion models for Bayesian inference in infinite dimensions

**Authors:** Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Solna, Maarten V. de Hoop

### Conformal PID Control for Time Series Prediction

**Authors:** Anastasios Angelopoulos, Emmanuel Candes, Ryan Tibshirani

### Consensus and Subjectivity of Skin Tone Annotation for ML Fairness

**Authors:** Candice Schumann, Gbolahan Olanubi, Auriel Wright, Ellis Monk, Courtney Heldreth, Susanna Ricco

### Consistent Aggregation of Objectives with Diverse Time Preferences Requires Non-Markovian Rewards

**Authors:** Silviu Pitis

### ContinuAR: Continuous Autoregression For Infinite-Fidelity Fusion

**Authors:** WEI XING, Yuxin Wang, Zheng Xing

### [Spotlight] Convergence of Adam Under Relaxed Assumptions

**Authors:** Haochuan Li, Alexander Rakhlin, Ali Jadbabaie

### Counting Distinct Elements Under Person-Level Differential Privacy

**Authors:** Thomas Steinke, Alexander Knop

### Covariance-adaptive best arm identification

**Authors:** El Mehdi Saad, Gilles Blanchard, Nicolas Verzelen

### Credal Marginal MAP

**Authors:** Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Fabio Cozman, Alexander Gray

### CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement

**Authors:** Qihe Huang, Lei Shen, Ruixin Zhang, Shouhong Ding, Binwu Wang, Zhengyang Zhou, Yang Wang

### [Spotlight] Curriculum Learning With Infant Egocentric Videos

**Authors:** Saber Sheybani, Himanshu Hansaria, Justin Wood, Linda Smith, Zoran Tiganj

### CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation

**Authors:** Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, Joyce Chai

### D$^2$CSG: Unsupervised Learning of Compact CSG Trees with Dual Complements and Dropouts

**Authors:** Fenggen Yu, Qimin Chen, Maham Tanveer, Ali Mahdavi Amiri, Hao Zhang

### D-Separation for Causal Self-Explanation

**Authors:** Wei Liu, Jun Wang, Haozhao Wang, Ruixuan Li, Zhiying Deng, YuanKai Zhang, Yang Qiu

### D4: Improving LLM Pretraining via Document De-Duplication and Diversification

**Authors:** Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos

### DICES Dataset: Diversity in Conversational AI Evaluation for Safety

**Authors:** Lora Aroyo, Alex Taylor, Mark Díaz, Christopher Homan, Alicia Parrish, Gregory Serapio-García, Vinodkumar Prabhakaran, Ding Wang

### DISCO-10M: A Large-Scale Music Dataset

**Authors:** Luca Lanzendörfer, Florian Grötschla, Emil Funke, Roger Wattenhofer

### DISCOVER: Making Vision Networks Interpretable via Competition and Dissection

**Authors:** Konstantinos Panousis, Sotirios Chatzis

### Data Portraits: Recording Foundation Model Training Data

**Authors:** Marc Marone, Benjamin Van Durme

### Data Pruning via Moving-one-Sample-out

**Authors:** Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi

### Data Selection for Language Models via Importance Resampling

**Authors:** Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang

### Datasets and Benchmarks for Nanophotonic Structure and Parametric Design Simulations

**Authors:** Jungtaek Kim, Mingxuan Li, Oliver Hinder, Paul Leu

### Decision Stacks: Flexible Reinforcement Learning via Modular Generative Models

**Authors:** Siyan Zhao, Aditya Grover

### Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory

**Authors:** Ankur Sikarwar, Mengmi Zhang

### Deep Equilibrium Based Neural Operators for Steady-State PDEs

**Authors:** Tanya Marwah, Ashwini Pokle, J. Zico Kolter, Zachary Lipton, Jianfeng Lu, Andrej Risteski

### Deep Momentum Multi-Marginal Schrödinger Bridge

**Authors:** Tianrong Chen, Guan-Horng Liu, Molei Tao, Evangelos Theodorou

### [Spotlight] Deep Neural Collapse Is Provably Optimal for the Deep Unconstrained Features Model

**Authors:** Peter Súkeník, Marco Mondelli, Christoph Lampert

### DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection

**Authors:** Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu

### Derandomized novelty detection with FDR control via conformal e-values

**Authors:** Meshi Bashari, Amir Epstein, Yaniv Romano, Matteo Sesia

### DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology

**Authors:** Marco Aversa, Gabriel Nobis, Miriam Hägele, Kai Standvoss, Mihaela Chirica, Roderick Murray-Smith, Ahmed Alaa, Lukas Ruff, Daniela Ivanova, Wojciech Samek, Frederick Klauschen, Bruno Sanguinetti, Luis Oala

### DiffPack: A Torsional Diffusion Model for Autoregressive Protein Side-Chain Packing

**Authors:** Yangtian Zhang, Zuobai Zhang, Bozitao Zhong, Sanchit Misra, Jian Tang

### DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model

**Authors:** Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, James Yu

### Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling of Tropical Cyclones

**Authors:** Asanobu Kitamoto, Jared Hwang, Bastien Vuillod, Lucas Gautier, Yingtao Tian, Tarin Clanuwat

### DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning

**Authors:** Alexander Liu, Heng-Jui Chang, Michael Auli, Wei-Ning Hsu, Jim Glass

### [Oral] Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, Chelsea Finn

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6B

### Discover and Align Taxonomic Context Priors  for Open-world Semi-Supervised Learning

**Authors:** Yu Wang, Zhun Zhong, Pengchong Qiao, Xuxin Cheng, Xiawu Zheng, Chang Liu, Nicu Sebe, Rongrong Ji, Jie Chen

### Dissecting Chain-of-Thought: Compositionality through In-Context Filtering and Learning

**Authors:** Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, Samet Oymak

### Distribution-Free Model-Agnostic Regression Calibration via Nonparametric Methods

**Authors:** Shang Liu, Zhongze Cai, Xiaocheng Li

### Diverse Community Data for Benchmarking Data Privacy Algorithms

**Authors:** Aniruddha Sen, Christine Task, Dhruv Kapur, Gary Howarth, Karan Bhagat

### Diverse Shape Completion via Style Modulated Generative Adversarial Networks

**Authors:** Wesley Khademi, Fuxin Li

### Do Not Marginalize Mechanisms, Rather Consolidate!

**Authors:** Moritz Willig, Matej Zečević, Devendra Dhami, Kristian Kersting

### Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework

**Authors:** Weiduo Liao, Ying Wei, Mingchen Jiang, Qingfu Zhang, Hisao Ishibuchi

### Does progress on ImageNet transfer to real-world datasets?

**Authors:** Alex Fang, Simon Kornblith, Ludwig Schmidt

### Doubly Constrained Fair Clustering

**Authors:** John Dickerson, Seyed Esmaeili, Jamie Morgenstern, Claire Jie Zhang

### Dream the Impossible: Outlier Imagination with Diffusion Models

**Authors:** Xuefeng Du, Yiyou Sun, Jerry Zhu, Yixuan Li

### [Spotlight] Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers

**Authors:** Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, Thomas Hofmann

### Dynamically Masked Discriminator for GANs

**Authors:** Wentian Zhang, Haozhe Liu, Bing Li, Jinheng Xie, Yawen Huang, Yuexiang Li, Yefeng Zheng, Bernard Ghanem

### E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning

**Authors:** Xiuhong Lin, Changjie Qiu, zhipeng cai, Siqi Shen, Yu Zang, Weiquan Liu, Xuesheng Bian, Matthias Müller, Cheng Wang

### ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram

**Authors:** Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon-myoung Kwon, Edward Choi

### EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models

**Authors:** Michael Wornow, Rahul Thapa, Ethan Steinberg, Jason Fries, Nigam Shah

### EPIC Fields: Marrying 3D Geometry and Video Understanding

**Authors:** Vadim Tschernezki, Ahmad Darkhalil, Zhifan Zhu, David Fouhey, Iro Laina, Diane Larlus, Dima Damen, Andrea Vedaldi

### Easy Bayesian Transfer Learning with Informative Priors

**Authors:** Martin Špendl, Klementina Pirc

### Efficient Robust Bayesian Optimization for Arbitrary Uncertain inputs

**Authors:** Lin Yang, Junlong Lyu, Wenlong Lyu, Zhitang Chen

### Efficient Symbolic Policy Learning with Differentiable Symbolic Expression

**Authors:** Jiaming Guo, Rui Zhang, Shaohui Peng, Qi Yi, Xing Hu, Ruizhi Chen, Zidong Du, xishan zhang, Ling Li, Qi Guo, Yunji Chen

### Ego4D Goal-Step: Toward Hierarchical Understanding of Procedural Activities

**Authors:** Yale Song, Eugene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, Lorenzo Torresani

### EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

**Authors:** Karttikeya Mangalam, Raiymbek Akshulakov, Jitendra Malik

### EgoTracks: A Long-term Egocentric Visual Object Tracking Dataset

**Authors:** Hao Tang, Kevin J Liang, Kristen Grauman, Matt Feiszli, Weiyao Wang

### Emergent Communication in Interactive Sketch Question Answering

**Authors:** Zixing Lei, Yiming Zhang, Yuxin Xiong, Siheng Chen

### Emergent Correspondence from Image Diffusion

**Authors:** Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, Bharath Hariharan

### [Spotlight] Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency

**Authors:** Owen Queen, Tom Hartvigsen, Teddy Koker, Huan He, Theodoros Tsiligkaridis, Marinka Zitnik

### Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization

**Authors:** Xilie Xu, Jingfeng ZHANG, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

### Enhancing Knowledge Transfer for Task Incremental Learning with Data-free Subnetwork

**Authors:** Qiang Gao, Xiaojun Shan, Yuchen Zhang, Fan Zhou

### Epidemic Learning: Boosting Decentralized Learning with Randomized Communication

**Authors:** Martijn De Vos, Sadegh Farhadkhani, Rachid Guerraoui, Anne-marie Kermarrec, Rafael Pires, Rishi Sharma

### Error Discovery By Clustering Influence Embeddings

**Authors:** Fulton Wang, Julius Adebayo, Sarah Tan, Diego Garcia-Olano, Narine Kokhlikyan

### Estimating Generic 3D Room Structures from 2D Annotations

**Authors:** Denys Rozumnyi, Stefan Popov, Kevis-kokitsi Maninis, Matthias Niessner, Vittorio Ferrari

### Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking

**Authors:** Juanhui Li, Harry Shomer, Haitao Mao, Shenglai Zeng, Yao Ma, Neil Shah, Jiliang Tang, Dawei Yin

### Evaluating Open-QA Evaluation

**Authors:** Cunxiang Wang, Sirui Cheng, Qipeng Guo, Yuanhao Yue, Bowen Ding, Zhikun Xu, Yidong Wang, Xiangkun Hu, Zheng Zhang, Yue Zhang

### Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning

**Authors:** Beichen Zhang, Kun Zhou, Xilin Wei, Xin Zhao, Jing Sha, Shijin Wang, Ji-Rong Wen

### Event Stream GPT: A Data Pre-processing and Modeling Library for Generative, Pre-trained Transformers over Continuous-time Sequences of Complex Events

**Authors:** Matthew McDermott, Bret Nestor, Peniel Argaw, Isaac S Kohane

### Exploiting Correlated Auxiliary Feedback in Parameterized Bandits

**Authors:** Arun Verma, Zhongxiang Dai, YAO SHU, Bryan Kian Hsiang Low

### [Spotlight] Exploring Geometry of Blind Spots in Vision models

**Authors:** Sriram Balasubramanian, Gaurang Sriramanan, Vinu Sankar Sadasivan, Soheil Feizi

### Exponential Lower Bounds for Fictitious Play in Potential Games

**Authors:** Ioannis Panageas, Nikolas Patris, Stratis Skoulakis, Volkan Cevher

### Expressivity-Preserving GNN Simulation

**Authors:** Fabian Jogl, Maximilian Thiessen, Thomas Gärtner

### Extremal Domain Translation with Neural Optimal Transport

**Authors:** Milena Gazdieva, Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev

### FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy

**Authors:** Zuhao Yang, Yingfang Yuan, Yang Xu, SHUO ZHAN, Huajun Bai, Kefan Chen

### FAMO: Fast Adaptive Multitask Optimization

**Authors:** Bo Liu, Yihao Feng, Peter Stone, Qiang Liu

### FAST: a Fused and Accurate Shrinkage Tree for Heterogeneous Treatment Effects Estimation

**Authors:** Jia Gu, Caizhi Tang, Han Yan, Qing Cui, Longfei Li, Jun Zhou

### FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning

**Authors:** Kun Song, Huimin Ma, Bochao Zou, Huishuai Zhang, Weiran Huang

### FELM: Benchmarking Factuality Evaluation of Large Language Models

**Authors:** shiqi chen, Yiran Zhao, Jinghan Zhang, I-Chun Chern, Siyang Gao, Pengfei Liu, Junxian He

### FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery

**Authors:** Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sebastien Giordano, boris Wattrelos

### FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding

**Authors:** Pengxiang Wu, Siman Wang, Kevin Dela Rosa, Derek Hu

### Fair Graph Distillation

**Authors:** Qizhang Feng, Zhimeng Jiang, Ruiquan Li, Yicheng Wang, Na Zou, Jiang Bian, Xia Hu

### FairLISA: Fair User Modeling with Limited Sensitive Attributes Information

**Authors:** zheng zhang, Qi Liu, Hao Jiang, Fei Wang, Yan Zhuang, Le Wu, Weibo Gao, Enhong Chen

### Fast Online Changepoint Detection via Functional Pruning CUSUM Statistics

**Authors:** Gaetano Romano, Idris A. Eckley, Paul Fearnhead, Guillem Rigaill

### Faster Discrete Convex Function Minimization with Predictions: The M-Convex Case

**Authors:** Taihei Oki, Shinsaku Sakaue

### Feature learning via mean-field Langevin dynamics: classifying sparse parities and beyond

**Authors:** Taiji Suzuki, Denny Wu, Kazusato Oko, Atsushi Nitanda

### Fed-FA: Theoretically Modeling Client Data Divergence for Federated Language Backdoor Defense

**Authors:** Zhiyuan Zhang, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun

### FedNAR: Federated Optimization with Normalized Annealing Regularization

**Authors:** Junbo Li, Ang Li, Chong Tian, Qirong Ho, Eric Xing, Hongyi Wang

### Federated Compositional Deep AUC Maximization

**Authors:** Xinwen Zhang, Yihan Zhang, Tianbao Yang, Richard Souvenir, Hongchang Gao

### Federated Learning with Bilateral Curation for Partially Class-Disjoint Data

**Authors:** Ziqing Fan, ruipeng zhang, Jiangchao Yao, Bo Han, Ya Zhang, Yanfeng Wang

### FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations

**Authors:** Chanakya Ekbote, Ajinkya Deshpande, Arun Iyer, SUNDARARAJAN SELLAMANICKAM, Ramakrishna Bairi

### Fitting trees to $\ell_1$-hyperbolic distances

**Authors:** Joon-Hyeok Yim, Anna Gilbert

### Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models

**Authors:** Haonan Duan, Adam Dziedzic, Nicolas Papernot, Franziska Boenisch

### Focus Your Attention when Few-Shot Classification

**Authors:** Haoqing Wang, Shibo Jie, Zhihong Deng

### ForecastPFN: Synthetically-Trained Zero-Shot Forecasting

**Authors:** Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha V Naidu, Colin White

### Fundamental Limits and Tradeoffs in Invariant Representation Learning

**Authors:** Han Zhao, Chen Dan, Bryon Aragam, Tommi Jaakkola, Geoffrey Gordon, Pradeep Ravikumar

### Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications

**Authors:** Xinyu Ma, Xu Chu, Yasha Wang, Yang Lin, Junfeng Zhao, Liantao Ma, Wenwu Zhu

### GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection

**Authors:** Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li

### GAUCHE: A Library for Gaussian Processes in Chemistry

**Authors:** Ryan-Rhys Griffiths, Leo Klarner, Henry Moss, Aditya Ravuri, Sang Truong, Yuanqi Du, Samuel Stanton, Gary Tom, Bojana Rankovic, Arian Jamasb, Aryan Deshwal, Julius Schwartz, Austin Tripp, Gregory Kell, Simon Frieder, Anthony Bourached, Alex Chan, Jacob Moss, Chengzhi Guo, Johannes Peter Dürholt, Saudamini Chaurasia, Ji Won Park, Felix Strieth-Kalthoff, Alpha Lee, Bingqing Cheng, Alan Aspuru-Guzik, Philippe Schwaller, Jian Tang

### GEO-Bench: Toward Foundation Models for Earth Monitoring

**Authors:** Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vazquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu

### GEX: A flexible method for approximating influence via Geometric Ensemble

**Authors:** SungYub Kim, Kyungsu Kim, Eunho Yang

### GLEMOS: Benchmark for Instantaneous Graph Learning Model Selection

**Authors:** Namyong Park, Ryan Rossi, Xing Wang, Antoine Simoulin, Nesreen K. Ahmed, Christos Faloutsos

### GenEval: An object-focused framework for evaluating text-to-image alignment

**Authors:** Dhruba Ghosh, Hannaneh Hajishirzi, Ludwig Schmidt

### Generalized Information-theoretic Multi-view Clustering

**Authors:** Weitian Huang, Sirui Yang, Hongmin Cai

### Generalized Weighted Path Consistency for Mastering Atari Games

**Authors:** Dengwei Zhao, Shikui Tu, Lei Xu

### Generating QM1B with PySCF$_{\text{IPU}}$

**Authors:** Alexander Mathiasen, Hatem Helal, Kerstin Klaser, Paul Balanca, Josef Dean, Carlo Luschi, Dominique Beaini, Andrew Fitzgibbon, Dominic Masters

### Generator Identification for Linear SDEs with Additive and Multiplicative Noise

**Authors:** Yuanyuan Wang, Xi Geng, Wei Huang, Biwei Huang, Mingming Gong

### GeoDE: a Geographically Diverse Evaluation Dataset for Object Recognition

**Authors:** Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron Adcock, Laurens van der Maaten, Deepti Ghadiyaram, Olga Russakovsky

### Gigastep - One Billion Steps per Second Multi-agent Reinforcement Learning

**Authors:** Mathias Lechner, lianhao yin, Tim Seyde, Tsun-Hsuan Johnson Wang, Wei Xiao, Ramin Hasani, Joshua Rountree, Daniela Rus

### Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity

**Authors:** Zhanpeng Zhou, Yongyi Yang, Xiaojiang Yang, Junchi Yan, Wei Hu

### Granger Components Analysis: Unsupervised learning of latent temporal dependencies

**Authors:** Jacek Dmochowski

### Graph Clustering with Graph Neural Networks

**Authors:** Anton Tsitsulin, John Palowitch, Bryan Perozzi, Emmanuel Müller

### Graph Denoising Diffusion for Inverse Protein Folding

**Authors:** Kai Yi, Bingxin Zhou, Yiqing Shen, Pietro Lió, Yuguang Wang

### Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis

**Authors:** Abhinav Nippani, Dongyue Li, Haotian Ju, Haris Koutsopoulos, Hongyang Zhang

### H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection

**Authors:** Yi Yu, Xue Yang, Qingyun Li, Yue Zhou, Feipeng Da, Junchi Yan

### HASSOD: Hierarchical Adaptive Self-Supervised Object Detection

**Authors:** Shengcao Cao, Dhiraj Joshi, Liangyan Gui, Yu-Xiong Wang

### HOH: Markerless Multimodal Human-Object-Human Handover Dataset with Large Object Count

**Authors:** Noah Wiederhold, Ava Megyeri, DiMaggio Paris, Sean Banerjee, Natasha Banerjee

### Hierarchical Multi-Agent Skill Discovery

**Authors:** Mingyu Yang, Yaodong Yang, Zhenbo Lu, Wengang Zhou, Houqiang Li

### [Spotlight] High-dimensional Asymptotics of Denoising Autoencoders

**Authors:** Hugo Cui, Lenka Zdeborová

### Higher-Order Uncoupled Dynamics Do Not Lead to Nash Equilibrium - Except When They Do

**Authors:** Sarah Toonsi, Jeff Shamma

### Hokoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks

**Authors:** Yun Qu, Boyuan Wang, Jianzhun Shao, Yuhang Jiang, Chen Chen, Zhenbin Ye, Liu Linc, Yang Feng, Lin Lai, Hongyang Qin, Minwen Deng, Juchao Zhuo, Deheng Ye, Qiang Fu, YANG GUANG, Wei Yang, Lanxiao Huang, Xiangyang Ji

### How hard are computer vision datasets? Calibrating dataset difficulty to viewing time

**Authors:** David Mayo, Jesse Cummings, Xinyu Lin, Dan Gutfreund, Boris Katz, Andrei Barbu

### How to Data in Datathons

**Authors:** Carlos Mougan, Richard Plant, Clare Teng, Marya Bazzi, Alvaro Cabrejas Egea, Ryan Chan, David Salvador Jasin, Martin Stoffel, Kirstie Whitaker, JULES MANSER

### HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face

**Authors:** Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang

### Human-Guided Complexity-Controlled Abstractions

**Authors:** Andi Peng, Mycal Tucker, Eoin Kenny, Noga Zaslavsky, Pulkit Agrawal, Julie A Shah

### Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context

**Authors:** Julian Tanke, Oh-Hun Kwon, Felix B Mueller, Andreas Doering, Jürgen Gall

### Hyper-HMM: aligning human brains and semantic features in a common latent event space

**Authors:** Caroline Lee, Jane Han, Ma Feilong, Guo Jiahui, James Haxby, Christopher Baldassano

### Hyper-Skin: A Hyperspectral Dataset for Reconstructing Facial Skin-Spectra from RGB Images

**Authors:** Pai Chet Ng, Zhixiang Chi, Yannick Verdie, Juwei Lu, Konstantinos N Plataniotis

### INSPECT: A Multimodal Dataset for Patient Outcome Prediction of Pulmonary Embolisms

**Authors:** Shih-Cheng Huang, Zepeng Huo, Ethan Steinberg, Chia-Chun Chiang, Curtis Langlotz, Matthew Lungren, Serena Yeung, Nigam Shah, Jason Fries

### Ignorance is Bliss: Robust Control via Information Gating

**Authors:** Manan Tomar, Riashat Islam, Matthew Taylor, Sergey Levine, Philip Bachman

### [Oral] Image Captioners Are Scalable Vision Learners Too

**Authors:** Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, Lucas Beyer

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6C

### ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification

**Authors:** Mohammad Reza Taesiri, Giang Nguyen, Sarra Habchi, Cor-Paul Bezemer, Anh Nguyen

### Imagine the Unseen World: A Benchmark for Systematic Generalization in Visual World Models

**Authors:** Yeongbin Kim, Gautam Singh, Junyeong Park, Caglar Gulcehre, Sungjin Ahn

### Implicit variance regularization in non-contrastive SSL

**Authors:** Manu Srinath Halvagal, Axel Laborieux, Friedemann Zenke

### [Oral] Improved Algorithms for Stochastic Linear Bandits Using Tail Bounds for Martingale Mixtures

**Authors:** Hamish Flynn, David Reeb, Melih Kandemir, Jan Peters

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6D

### Improving *day-ahead* Solar Irradiance Time Series Forecasting by Leveraging Spatio-Temporal Context

**Authors:** Oussama Boussif, Ghait Boukachab, Dan Assouline, Stefano Massaroli, Tianle Yuan, Loubna Benabbou, Yoshua Bengio

### Improving Self-supervised Molecular Representation Learning using Persistent Homology

**Authors:** Yuankai Luo, Lei Shi, Veronika Thost

### Inference for Gaussian Processes with Matern Covariogram on Compact Riemannian Manifolds

**Authors:** Didong Li, Wenpin Tang, Sudipto Banerjee

### InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion

**Authors:** Fangzhou Lin, Yun Yue, Ziming Zhang, Songlin Hou, Kazunori Yamada, Vijaya Kolachalama, Venkatesh Saligrama

### InstanT: Semi-supervised Learning with Instance-dependent Thresholds

**Authors:** Muyang Li, Runze Wu, Haoyu Liu, Jun Yu, Xun Yang, Bo Han, Tongliang Liu

### InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

**Authors:** Wenliang Dai, Junnan Li, DONGXU LI, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, Steven Hoi

### Intelligent Knee Sleeves: A Real-time Multimodal Dataset for 3D Lower Body Motion Estimation Using Smart Textile

**Authors:** Wenwen Zhang, Arvin Tashakori, Zenan Jiang, Amir Servati, Harishkumar Narayana, Saeid Soltanian, Rou Yi Yeap, Menghan Ma, Lauren Toy, Peyman Servati

### InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback

**Authors:** John Yang, Akshara Prabhakar, Karthik Narasimhan, Shunyu Yao

### Interactive Visual Reasoning under Uncertainty

**Authors:** Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu

### Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction

**Authors:** Ruoyu Li, Qing Li, Yu Zhang, Dan Zhao, Yong Jiang, Yong Yang

### Into the LAION’s Den: Investigating Hate in Multimodal Datasets

**Authors:** Abeba Birhane, vinay prabhu, Sanghyun Han, Vishnu Boddeti, Sasha Luccioni

### Into the Single Cell Multiverse: an End-to-End Dataset for Procedural Knowledge Extraction in Biomedical Texts

**Authors:** Ruth Dannenfelser, Jeffrey Zhong, Ran Zhang, Vicky Yao

### Intrinsic Gaussian Process on Unknown Manifolds with Probabilistic Metrics

**Authors:** mu niu, Zhenwen Dai, Pokman Cheung, Yizhu Wang

### Is RLHF More Difficult than Standard RL? A Theoretical Perspective

**Authors:** Yuanhao Wang, Qinghua Liu, Chi Jin

### [Oral] Jailbroken: How Does LLM Safety Training Fail?

**Authors:** Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6A

### Knowledge Distillation Performs Partial Variance Reduction

**Authors:** Mher Safaryan, Alexandra Peste, Dan Alistarh

### Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks

**Authors:** Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, Sung Ju Hwang

### Knowledge-based in silico models and dataset for the comparative evaluation of mammography AI for a range of breast characteristics, lesion conspicuities and doses

**Authors:** Elena Sizikova, Niloufar Saharkhiz, Diksha Sharma, Miguel Lago, Berkman Sahiner, Jana Delfino, Aldo Badano

### Kullback-Leibler Maillard Sampling for Multi-armed Bandits with Bounded Rewards

**Authors:** Hao Qin, Kwang-Sung Jun, Chicheng Zhang

### LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark

**Authors:** Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiyong Wang, Lu Sheng, LEI BAI, Jing Shao, Wanli Ouyang

### LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

**Authors:** Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, Jianfeng Gao

### LOVM: Language-Only Vision Model Selection

**Authors:** Orr Zohar, Shih-Cheng Huang, Kuan-Chieh Wang, Serena Yeung

### LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching

**Authors:** Duy M. H. Nguyen, Hoang Nguyen, Nghiem Diep, Tan Ngoc Pham, Tri Cao, Binh Nguyen, Paul Swoboda, Nhat Ho, Shadi Albarqouni, Pengtao Xie, Daniel Sonntag, Mathias Niepert

### LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

**Authors:** Artur Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, Nikolaus Adams

### Language-driven Scene Synthesis using Multi-conditional Diffusion Model

**Authors:** An Dinh Vuong, Minh Nhat VU, Toan Nguyen, Baoru Huang, Dzung Nguyen, Thieu Vo, Anh Nguyen

### Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias

**Authors:** Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang

### Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset

**Authors:** Saeid Alavi Naeini, Raeid Saqur, Mozhgan Saeidi, John Giorgi, Babak Taati

### Latent Graph Inference with Limited Supervision

**Authors:** Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, Yun Fu

### Learning Causal Models under Independent Changes

**Authors:** Sarah Mameche, David Kaltenpoth, Jilles Vreeken

### Learning Curves for Noisy Heterogeneous Feature-Subsampled Ridge Ensembles

**Authors:** Ben Ruben, Cengiz Pehlevan

### Learning Human Action Recognition Representations Without Real Humans

**Authors:** Howard Zhong, Samarth Mishra, Donghyun Kim, SouYoung Jin, Rameswar Panda, Hilde Kuehne, Leonid Karlinsky, Venkatesh Saligrama, Aude Oliva, Rogerio Feris

### Learning a 1-layer conditional generative model in total variation

**Authors:** Ajil Jalal, Justin Kang, Ananya Uppal, Kannan Ramchandran, Eric Price

### Learning a Neuron by a Shallow ReLU Network: Dynamics and Implicit Bias for Correlated Inputs

**Authors:** Dmitry Chistikov, Matthias Englert, Ranko Lazic

### Learning from Rich Semantics and Coarse Locations for Long-tailed Object Detection

**Authors:** Lingchen Meng, Xiyang Dai, Jianwei Yang, Dongdong Chen, Yinpeng Chen, Mengchen Liu, Yi-Ling Chen, Zuxuan Wu, Lu Yuan, Yu-Gang Jiang

### Learning to Taste: A Multimodal Wine Dataset

**Authors:** Thoranna Bender, Simon Sørensen, Alireza Kashani, Kristjan Hjorleifsson, Grethe Hyldig, Søren Hauberg, Serge Belongie, Frederik Warburg

### Lending Interaction Wings to Recommender Systems with Conversational Agents

**Authors:** Jiarui Jin, Xianyu Chen, Fanghua Ye, Mengyue Yang, Yue Feng, Weinan Zhang, Yong Yu, Jun Wang

### [Spotlight] Let the Flows Tell:  Solving Graph Combinatorial Problems with GFlowNets

**Authors:** Dinghuai Zhang, Hanjun Dai, Nikolay Malkin, Aaron Courville, Yoshua Bengio, Ling Pan

### Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis

**Authors:** Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Taesu Kim, Daehyun Ahn, Hyungjun Kim, jae-joon kim

### LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing

**Authors:** Su Zheng, Haoyu Yang, Binwu Zhu, Bei Yu, Martin Wong

### Lo-Hi: Practical ML Drug Discovery Benchmark

**Authors:** Simon Steshin

### Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration

**Authors:** Fenja Falta, Christoph Großbröhmer, Alessa Hering, Alexander Bigalke, Mattias Heinrich

### M$^2$Hub: Unlocking the Potential of Machine Learning for Materials Discovery

**Authors:** Yuanqi Du, Yingheng Wang, Yining Huang, Jianan Canal Li, Yanqiao Zhu, Tian Xie, Chenru Duan, John Gregoire, Carla Gomes

### M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models

**Authors:** Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, Lidong Bing

### MADG: Margin-based Adversarial Learning for Domain Generalization

**Authors:** Aveen Dayal, Vimal K B, Linga Reddy Cenkeramaddi, C Mohan, Abhinav Kumar, Vineeth N Balasubramanian

### MADLAD-400: A Multilingual And Document-Level Large Audited Dataset

**Authors:** Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat

### MARBLE: Music Audio Representation Benchmark for Universal Evaluation

**Authors:** Ruibin Yuan, Yinghao Ma, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, zhuo le, Yiqi Liu, Jiawen Huang, Zeyue Tian, Binyue Deng, Ningzhi Wang, Chenghua Lin, Emmanouil Benetos, Anton Ragni, Norbert Gyenge, Roger Dannenberg, wenhu chen, Gus Xia, Wei Xue, Si Liu, Shi Wang, Ruibo Liu, Yike Guo, Jie Fu

### MLFMF: Data Sets for Machine Learning for Mathematical Formalization

**Authors:** Andrej Bauer, Matej Petković, Ljupco Todorovski

### MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing

**Authors:** Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, Lihua Xie

### MMD Aggregated Two-Sample Test

**Authors:** Antonin Schrab, Ilmun Kim, Mélisande Albert, Béatrice Laurent, Benjamin Guedj, Arthur Gretton

### [Spotlight] MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion

**Authors:** Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, Yasutaka Furukawa

### MVDoppler: Unleashing the Power of Multi-View Doppler for MicroMotion-based Gait Classification

**Authors:** Soheil Hor, Shubo Yang, Jaeho Choi, Amin Arbabian

### MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing

**Authors:** Kai Zhang, Lingbo Mo, wenhu chen, Huan Sun, Yu Su

### Massively Multilingual Corpus of Sentiment Datasets and Multi-faceted Sentiment Classification Benchmark

**Authors:** Lukasz Augustyniak, Szymon Woźniak, Marcin Gruza, Piotr Gramacki, Krzysztof Rajda, Mikołaj Morzy, Tomasz Kajdanowicz

### [Spotlight] Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness

**Authors:** Gang Li, Wei Tong, Tianbao Yang

### Mechanic: A Learning Rate Tuner

**Authors:** Ashok Cutkosky, Aaron Defazio, Harsh Mehta

### [Spotlight] Mechanism Design for Collaborative Normal Mean Estimation

**Authors:** Yiding Chen, Jerry Zhu, Kirthevasan Kandasamy

### MedSat: A Public Health Dataset for England Featuring Medical Prescriptions and Satellite Imagery

**Authors:** Sanja Scepanovic, Ivica Obadic, Sagar Joglekar, Laura GIUSTARINI, Cristiano Nattero, Daniele Quercia, Xiaoxiang Zhu

### Meta-in-context learning in large language models

**Authors:** Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt Botvinick, Jane Wang, Eric Schulz

### [Oral] MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

**Authors:** Zeyuan Ma, Hongshu Guo, Jiacheng Chen, Zhenrui Li, Guojun Peng, Yue-Jiao Gong, Yining Ma, Zhiguang Cao

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6B

### MiliPoint: A Point Cloud Dataset for mmWave Radar

**Authors:** Han Cui, Shu Zhong, Jiacheng Wu, Zichao Shen, Naim Dahnoun, Yiren Zhao

### Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension

**Authors:** Moritz Haas, David Holzmüller, Ulrike Luxburg, Ingo Steinwart

### Mind2Web: Towards a Generalist Agent for the Web

**Authors:** Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, Yu Su

### Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks

**Authors:** Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, J Terry

### Minimax Forward and Backward Learning of Evolving Tasks with Performance Guarantees

**Authors:** Veronica Alvarez, Santiago Mazuelas, Jose A. Lozano

### Minimax Optimal Rate for Parameter Estimation in Multivariate Deviated Models

**Authors:** Dat Do, Huy Nguyen, Khai Nguyen, Nhat Ho

### Modeling Human Visual Motion Processing with Trainable Motion Energy Sensing and a Self-attention Network

**Authors:** Zitang Sun, Yen-Ju Chen, Yung-Hao Yang, Shin'ya Nishida

### Molecule Joint Auto-Encoding: Trajectory Pretraining with 2D and 3D Diffusion

**Authors:** weitao Du, Jiujiu Chen, Xuecang Zhang, Zhi-Ming Ma, Shengchao Liu

### Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset

**Authors:** Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, Lei Zhang

### Multi-scale Diffusion Denoised Smoothing

**Authors:** Jongheon Jeong, Jinwoo Shin

### MultiVENT: Multilingual Videos of Events and Aligned Natural Text

**Authors:** Kate Sanders, David Etter, Reno Kriz, Benjamin Van Durme

### Multimodal Clinical Benchmark for Emergency Care (MC-BEC): A Comprehensive Benchmark for Evaluating Foundation Models in Emergency Medicine

**Authors:** Emma Chen, Aman Kansal, Julie Chen, Boyang Tom Jin, Julia Reisler, David Kim, Pranav Rajpurkar

### NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations

**Authors:** Varun Jampani, Kevis-kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, Howard Zhou

### NIS3D: A Completely Annotated Benchmark for Dense 3D Nuclei Image Segmentation

**Authors:** Wei Zheng, Cheng Peng, Zeyuan Hou, Boyu Lyu, Mengfan Wang, Xuelong Mi, Shuoxuan Qiao, Yinan Wan, Guoqiang Yu

### Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation

**Authors:** Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, P. R. Kumar, Chao Tian

### NetHack is Hard to Hack

**Authors:** Ulyana Piterbarg, Lerrel Pinto, Rob Fergus

### Network Regression with Graph Laplacians

**Authors:** Yidong Zhou, Hans-Georg Müller

### Neural Algorithmic Reasoning Without Intermediate Supervision

**Authors:** Gleb Rodionov, Liudmila Prokhorenkova

### Neural Ideal Large Eddy Simulation: Modeling Turbulence with Neural Stochastic Differential Equations

**Authors:** Anudhyan Boral, Zhong Yi Wan, Leonardo Zepeda-Núñez, James Lottes, Qing Wang, Yi-Fan Chen, John Anderson, Fei Sha

### Neural MMO 2.0: A Massively Multi-task Addition to Massively Multi-agent Learning

**Authors:** Joseph Suarez, David Bloomin, Kyoung Whan Choe, Hao Xiang Li, Ryan Sullivan, Nishaanth Kanna, Daniel Scott, Rose Shuman, Herbie Bradley, Louis Castricato, Phillip Isola, Chenghui Yu, Yuhao Jiang, Qimai Li, Jiaxin Chen, Xiaolong Zhu

### NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics

**Authors:** Anwar Said, Roza Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, Xenofon Koutsoukos

### New Complexity-Theoretic Frontiers of Tractability for Neural Network Training

**Authors:** Cornelius Brand, Robert Ganian, Mathis Rocton

### No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models

**Authors:** Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, Matt Kusner

### Non-Stationary Bandits with Auto-Regressive Temporal Dependency

**Authors:** Qinyi Chen, Negin Golrezaei, Djallel Bouneffouf

### Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts

**Authors:** Emanuele Marconato, Stefano Teso, Antonio Vergari, Andrea Passerini

### OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents

**Authors:** Hugo Laurençon, Lucile Saulnier, Leo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

### OBJECT 3DIT: Language-guided 3D-aware Image Editing

**Authors:** Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, Tanmay Gupta

### OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment

**Authors:** Yiheng Zhu, Yang Zhan, Xuankun Huang, Yuwei Chen, yujie Chen, Jiangwen Wei, Wei Feng, Yinzhi Zhou, Haoyuan Hu, Jieping Ye

### OV-PARTS: Towards Open-Vocabulary Part Segmentation

**Authors:** Meng Wei, Xiaoyu Yue, Wenwei Zhang, Shu Kong, Xihui Liu, Jiangmiao Pang

### Objaverse-XL: A Universe of 10M+ 3D Objects

**Authors:** Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, Ali Farhadi

### Object Reprojection Error (ORE): Camera pose benchmarks from lightweight tracking annotations

**Authors:** Xingyu Chen, Weiyao Wang, Hao Tang, Matt Feiszli

### Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities

**Authors:** Andrii Zadaianchuk, Maximilian Seitzer, Georg Martius

### Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving

**Authors:** Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao

### Offline RL with Discrete Proxy Representations for Generalizability in POMDPs

**Authors:** Pengjie Gu, Xinyu Cai, Dong Xing, Xinrun Wang, Mengchen Zhao, Bo An

### On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

**Authors:** Federico Errica

### On Evaluating Adversarial Robustness of Large Vision-Language Models

**Authors:** Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan LI, Ngai-Man (Man) Cheung, Min Lin

### On Sample-Efficient Offline Reinforcement Learning: Data Diversity, Posterior Sampling and Beyond

**Authors:** Thanh Nguyen-Tang, Raman Arora

### On the Importance of Exploration for Generalization in Reinforcement Learning

**Authors:** Yiding Jiang, J. Zico Kolter, Roberta Raileanu

### On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets

**Authors:** Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong

### On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective

**Authors:** Zeke Xie, Zhiqiang Xu, Jingzhao Zhang, Issei Sato, Masashi Sugiyama

### One-Line-of-Code Data Mollification Improves Optimization of Likelihood-based Generative Models

**Authors:** Ba-Hien Tran, Giulio Franzese, Pietro Michiardi, Maurizio Filippone

### [Spotlight] Online Constrained Meta-Learning: Provable Guarantees for Generalization

**Authors:** Siyuan Xu, Minghui Zhu

### Online Corrupted User Detection and Regret Minimization

**Authors:** Zhiyong Wang, Jize Xie, Tong Yu, Shuai Li, John C.S. Lui

### [Spotlight] Online List Labeling with Predictions

**Authors:** Samuel McCauley, Ben Moseley, Aidin Niaparast, Shikha Singh

### Online robust non-stationary estimation

**Authors:** Abishek Sankararaman, Balakrishnan Narayanaswamy

### OpenAGI: When LLM Meets Domain Experts

**Authors:** Yingqiang Ge, Wenyue Hua, Kai Mei, jianchao ji, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang

### OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping

**Authors:** Huijie Wang, Tianyu Li, Yang Li, Li Chen, Chonghao Sima, Zhenbo Liu, Bangjun Wang, Peijin Jia, Yuting Wang, Shengyin Jiang, Feng Wen, Hang Xu, Ping Luo, Junchi Yan, Wei Zhang, Hongyang Li

### OpenProteinSet: Training data for structural biology at scale

**Authors:** Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Dan Berenberg, Ian Fisk, Andrew Watkins, Stephen Ra, Richard Bonneau, Mohammed AlQuraishi

### OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning

**Authors:** Cheng Tan, Siyuan Li, Zhangyang Gao, Wenfei Guan, Zedong Wang, Zicheng Liu, Lirong Wu, Stan Z. Li

### Operation-Level Early Stopping for Robustifying Differentiable NAS

**Authors:** Shen Jiang, Zipeng Ji, Guanghui Zhu, Chunfeng Yuan, Yihua Huang

### Optimal Algorithms for the Inhomogeneous Spiked Wigner Model

**Authors:** Aleksandr Pak, Justin Ko, Florent Krzakala

### Optimal Block-wise Asymmetric Graph Construction for Graph-based Semi-supervised Learning

**Authors:** Zixing Song, Yifei Zhang, Irwin King

### [Oral] Optimal Learners for Realizable Regression: PAC Learning and Online Learning

**Authors:** Idan Attias, Steve Hanneke, Alkis Kalavasis, Amin Karbasi, Grigoris Velegkas

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6D

### Optimal Regret Is Achievable with Bounded Approximate Inference Error: An Enhanced Bayesian Upper Confidence Bound Framework

**Authors:** Ziyi Huang, Henry Lam, Amirhossein Meisami, Haofeng Zhang

### Optimal Treatment Regimes for Proximal Causal Learning

**Authors:** Tao Shen, Yifan Cui

### Optimal approximation using complex-valued neural networks

**Authors:** Paul Geuchen, Felix Voigtlaender

### Optimality of Message-Passing Architectures for Sparse Graphs

**Authors:** Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath

### [Spotlight] Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework  for Online RL

**Authors:** Qinghua Liu, Gellert Weisz, András György, Chi Jin, Csaba Szepesvari

### Optimized Covariance Design for AB Test on Social Network under Interference

**Authors:** Qianyi Chen, Bo Li, Lu Deng, Yong Wang

### Outlier-Robust Wasserstein DRO

**Authors:** Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

### P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting

**Authors:** Sungwon Kim, Kevin Shih, rohan badlani, Joao Felipe Santos, Evelina Bakhturina, Mikyas Desta, Rafael Valle, Sungroh Yoon, Bryan Catanzaro

### PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization

**Authors:** Jiancong Xiao, Ruoyu Sun, Zhi-Quan Luo

### PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection

**Authors:** Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, Hao Zhao

### PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance

**Authors:** Peiqing Yang, Shangchen Zhou, Qingyi Tao, Chen Change Loy

### PPi: Pretraining Brain Signal Model for Patient-independent Seizure Detection

**Authors:** Zhizhang Yuan, Daoze Zhang, YANG YANG, Junru Chen, Yafeng Li

### PTADisc: A Cross-Course Dataset Supporting Personalized Learning in Cold-Start Scenarios

**Authors:** Liya Hu, Zhiang Dong, Jingyuan Chen, Guifeng Wang, Zhihua Wang, Zhou Zhao, Fei Wu

### PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning

**Authors:** Florian Bordes, Shashank Shekhar, Mark Ibrahim, Diane Bouchacourt, Pascal Vincent, Ari Morcos

### Pairwise GUI Dataset Construction Between Android Phones and Tablets

**Authors:** han hu, Haolan Zhan, Yujin Huang, Di Liu

### Parallel-mentoring for Offline Model-based Optimization

**Authors:** Can Chen, Christopher Beckham, Zixuan Liu, Xue (Steve) Liu, Chris Pal

### Perception Test: A Diagnostic Benchmark for Multimodal Video Models

**Authors:** Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, joseph heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, Joao Carreira

### Personalized Dictionary Learning for Heterogeneous Datasets

**Authors:** Geyu Liang, Naichen Shi, Raed AL Kontar, Salar Fattahi

### Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning

**Authors:** Sotetsu Koyamada, Shinri Okano, Soichiro Nishimori, Yu Murata, Keigo Habara, Haruka Kita, Shin Ishii

### Physics-Informed Bayesian Optimization of Variational Quantum Circuits

**Authors:** Kim Nicoli, Christopher J. Anders, Lena Funcke, Tobias Hartung, Karl Jansen, Stefan Kühn, Klaus-Robert Müller, Paolo Stornati, Pan Kessel, Shinichi Nakajima

### Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties

**Authors:** Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel Bear, Chuang Gan, Josh Tenenbaum, Dan Yamins, Judith Fan, Kevin Smith

### Pitfall of Optimism: Distributional Reinforcement Learning by Randomizing Risk Criterion

**Authors:** Taehyun Cho, Seungyub Han, Heesoo Lee, Kyungjae Lee, Jungwoo Lee

### PlanE: Representation Learning over Planar Graphs

**Authors:** Radoslav Dimitrov, Zeyang Zhao, Ralph Abboud, Ismail Ceylan

### Practical Equivariances via Relational Conditional Neural Processes

**Authors:** Daolang Huang, Manuel Haussmann, Ulpu Remes, ST John, Grégoire Clarté, Kevin Sebastian Luck, Samuel Kaski, Luigi Acerbi

### [Spotlight] Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers

**Authors:** Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, David Pan

### PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning

**Authors:** Neeratyoy Mallik, Edward Bergman, Carl Hvarfner, Danny Stoll, Maciej Janowski, Marius Lindauer, Luigi Nardi, Frank Hutter

### Probabilistic inverse optimal control for non-linear partially observable systems disentangles perceptual uncertainty and behavioral costs

**Authors:** Dominik Straub, Matthias Schultheis, Heinz Koeppl, Constantin Rothkopf

### Progressive Ensemble Distillation: Building Ensembles for Efficient Inference

**Authors:** Don Dennis, Abhishek Shetty, Anish Prasad Sevekari, Kazuhito Koishida, Virginia Smith

### ProteinInvBench: Benchmarking Protein Inverse Folding on Diverse Tasks, Models, and Metrics

**Authors:** Zhangyang Gao, Cheng Tan, Yijie Zhang, Xingran Chen, Lirong Wu, Stan Z. Li

### ProteinShake: Building datasets and benchmarks for deep learning on protein structures

**Authors:** Tim Kucera, Carlos Oliver, Dexiong Chen, Karsten Borgwardt

### [Spotlight] Provable benefits of score matching

**Authors:** Chirag Pabbaraju, Dhruv Rohatgi, Anish Prasad Sevekari, Holden Lee, Ankur Moitra, Andrej Risteski

### [Spotlight] Proximity-Informed Calibration for Deep Neural Networks

**Authors:** Miao Xiong, Ailin Deng, Pang Wei Koh, Jiaying Wu, Shen Li, Jianqing Xu, Bryan Hooi

### QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data

**Authors:** Simone Papicchio, Paolo Papotti, Luca Cagliero

### [Spotlight] QuIP: 2-Bit Quantization of Large Language Models With Guarantees

**Authors:** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa

### QuadAttac$K$: A Quadratic Programming Approach to Learning Ordered Top-$K$ Adversarial Attacks

**Authors:** Thomas Paniagua, Ryan Grainger, Tianfu Wu

### Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

**Authors:** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort

### Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond

**Authors:** Anna Hedström, Leander Weber, Daniel Krakowczyk, Dilyara Bareeva, Franz Motzkus, Wojciech Samek, Sebastian Lapuschkin, Marina Höhne

### [Spotlight] Quasi-Monte Carlo Graph Random Features

**Authors:** Isaac Reid, Adrian Weller, Krzysztof M Choromanski

### RELIC: Reproducibility and Extension on LIC metric in quantifying bias in captioning models

**Authors:** Martijn van Raaphorst, Egoitz Gonzalez, Marta Grasa, Paula Antequera Hernández

### RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments

**Authors:** Mengxue Qu, Yu Wu, Wu Liu, Xiaodan Liang, Jingkuan Song, Yao Zhao, Yunchao Wei

### RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization

**Authors:** Zhecheng Yuan, Sizhe Yang, Pu Hua, Can Chang, Kaizhe Hu, Huazhe Xu

### RL-based Stateful Neural Adaptive Sampling and Denoising for Real-Time Path Tracing

**Authors:** Antoine Scardigli, Lukas Cavigelli, Lorenz K. Müller

### RVD: A Handheld Device-Based Fundus Video Dataset for Retinal Vessel Segmentation

**Authors:** MD WAHIDUZZAMAN KHAN, Hongwei Sheng, Hu Zhang, Heming Du, Sen Wang, Minas Coroneo, Farshid Hajati, Sahar Shariflou, Michael Kalloniatis, Jack Phu, Ashish Agar, Zi Huang, S.Mojtaba Golzan, Xin Yu

### [Oral] Random Cuts are Optimal for Explainable k-Medians

**Authors:** Konstantin Makarychev, Liren Shan

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6D

### ReDS: Offline RL With Heteroskedastic Datasets via Support Constraints

**Authors:** Anikait Singh, Aviral Kumar, Quan Vuong, Yevgen Chebotar, Sergey Levine

### [Spotlight] Real-World Image Variation by Aligning Diffusion Inversion Chain

**Authors:** Yuechen Zhang, Jinbo Xing, Eric Lo, Jiaya Jia

### RealTime QA: What's the Answer Right Now?

**Authors:** Jungo Kasai, Keisuke Sakaguchi, yoichi takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah Smith, Yejin Choi, Kentaro Inui

### Realistic Synthetic Financial Transactions for Anti-Money Laundering Models

**Authors:** Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Beni Egressy, Andreea Anghel, Kubilay Atasu

### Recasting Continual Learning as Sequence Modeling

**Authors:** Soochan Lee, Jaehyeon Son, Gunhee Kim

### Reflexion: language agents with verbal reinforcement learning

**Authors:** Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao

### Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark

**Authors:** Lasse Hansen, Nabeel Seedat, Mihaela van der Schaar, Andrija Petrovic

### Replicable Reinforcement Learning

**Authors:** Eric Eaton, Marcel Hussing, Michael Kearns, Jessica Sorrell

### Reproducibility Study of “Quantifying Societal Bias Amplification in Image Captioning”

**Authors:** Farrukh Baratov, Goksenin Yuksel, Darie Petcu, Jan Bakker

### Reproducibility Study of ”Label-Free Explainability for Unsupervised Models”

**Authors:** Julius Wagenbach, Gergely Papp, Niklas Mather, Laurens de Vries

### Reproducibility study of the Fairness-enhanced Node Representation Learning

**Authors:** Gijs Moens, Job De Witte, Tobias Gobel, Meggie Van den Oever

### Residual Alignment: Uncovering the Mechanisms of Residual Networks

**Authors:** Jianing Li, Vardan Papyan

### Restart Sampling for Improving Generative Processes

**Authors:** Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, Tommi Jaakkola

### Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?

**Authors:** Fan Yao, Chuanhao Li, Karthik Abinav Sankararaman, Yiming Liao, Yan Zhu, Qifan Wang, Hongning Wang, Haifeng Xu

### Retrieval-Augmented Multiple Instance Learning

**Authors:** Yufei CUI, Ziquan Liu, Yixin Chen, Yuchen Lu, Xinyue Yu, Xue (Steve) Liu, Tei-Wei Kuo, Miguel Rodrigues, Chun Jason XUE, Antoni Chan

### Revealing the unseen: Benchmarking video action recognition under occlusion

**Authors:** Shresth Grover, Vibhav Vineet, Yogesh Rawat

### Revisiting Adversarial Robustness Distillation from the Perspective of Robust Fairness

**Authors:** Xinli Yue, Mou Ningping, Qian Wang, Lingchen Zhao

### Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union

**Authors:** Zifu Wang, Maxim Berman, Amal Rannen-Triki, Philip Torr, Devis Tuia, Tinne Tuytelaars, Luc V Gool, Jiaqian Yu, Matthew Blaschko

### Revisiting the Evaluation of Image Synthesis with GANs

**Authors:** mengping yang, Ceyuan Yang, Yichi Zhang, Qingyan Bai, Yujun Shen, Bo Dai

### Revisiting the Minimalist Approach to Offline Reinforcement Learning

**Authors:** Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, Sergey Kolesnikov

### [Spotlight] Rewiring Neurons in Non-Stationary Environments

**Authors:** Zhicheng Sun, Yadong Mu

### Riemannian SAM: Sharpness-Aware Minimization on Riemannian Manifolds

**Authors:** Jihun Yun, Eunho Yang

### RoboHive: A Unified Framework for Robot Learning

**Authors:** Vikash Kumar, Rutav Shah, Gaoyue Zhou, Vincent Moens, Vittorio Caggiano, Abhishek Gupta, Aravind Rajeswaran

### Robust Bayesian Satisficing

**Authors:** Artun Saday, Y. Cahit Yıldırım, Cem Tekin

### Robust Data Valuation with Weighted Banzhaf Values

**Authors:** Weida Li, Yaoliang Yu

### [Spotlight] Robust Model Reasoning and Fitting via Dual Sparsity Pursuit

**Authors:** Xingyu Jiang, Jiayi Ma

### SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model

**Authors:** Di Wang, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao, Liangpei Zhang

### SARAMIS: Simulation Assets for Robotic Assisted and Minimally Invasive Surgery

**Authors:** Nina Montana-Brown, Shaheer U. Saeed, Ahmed Abdulaal, Thomas Dowrick, Yakup Kilic, Sophie Wilkinson, Jack Gao, Meghavi Mashar, Chloe He, Alkisti Stavropoulou, Emma Thomson, Zachary MC Baum, Simone Foti, Brian Davidson, Yipeng Hu, Matthew Clarkson

### SEVA: Leveraging sketches to evaluate alignment between human and machine visual abstraction

**Authors:** Kushin Mukherjee, Holly Huey, Xuanchen Lu, Yael Vinker, Rio Aguina-Kang, Ariel Shamir, Judith Fan

### SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark

**Authors:** Zeyu Zhang, Robert Pless, Nadia Shakoor, Austin Carnahan, Abby Stylianou

### SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning

**Authors:** Benjamin Ellis, Jonathan Cook, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob Foerster, Shimon Whiteson

### SODA: Robust Training of Test-Time Data Adaptors

**Authors:** Zige Wang, Yonggang Zhang, Zhen Fang, Long Lan, Wenjing Yang, Bo Han

### SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning

**Authors:** Yi-Chung Chen, Hsi-Wen Chen, Shun-Gui Wang, Ming-syan Chen

### STARSS23: An Audio-Visual Dataset of Spatial Recordings of Real Scenes with Spatiotemporal Annotations of Sound Events

**Authors:** Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel A. Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, Tuomas Virtanen, Yuki Mitsufuji

### SafeDICE: Offline Safe Imitation Learning with Non-Preferred Demonstrations

**Authors:** Youngsoo Jang, Geon-Hyeong Kim, Jongmin Lee, Sungryull Sohn, Byoungjip Kim, Honglak Lee, Moontae Lee

### Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark

**Authors:** Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang

### Sample Complexity Bounds for Score-Matching: Causal Discovery and Generative Modeling

**Authors:** Zhenyu Zhu, Francesco Locatello, Volkan Cevher

### Sample-efficient Multi-objective Molecular Optimization with GFlowNets

**Authors:** Yiheng Zhu, Jialu Wu, Chaowen Hu, Jiahuan Yan, kim hsieh, Tingjun Hou, Jian Wu

### Sampling weights of deep neural networks

**Authors:** Erik L Bolager, Iryna Burak, Chinmay Datar, Qing Sun, Felix Dietrich

### SatBird: a Dataset for Bird Species Distribution Modeling using Remote Sensing and Citizen Science Data

**Authors:** Mélisande Teng, Amna Elmustafa, Benjamin Akera, Yoshua Bengio, Hager Radi, Hugo Larochelle, David Rolnick

### Scalable 3D Captioning with Pretrained Models

**Authors:** Tiange Luo, Chris Rockwell, Honglak Lee, Justin Johnson

### Scalable Fair Influence Maximization

**Authors:** Xiaobin Rui, Zhixiao Wang, Jiayu Zhao, Lichao Sun, Wei Chen

### Scalable Membership Inference Attacks via Quantile Regression

**Authors:** Martin Bertran, Shuai Tang, Aaron Roth, Michael Kearns, Jamie Morgenstern, Steven Wu

### Scientific Document Retrieval using Multi-level Aspect-based Queries

**Authors:** Jianyou (Andre) Wang, Kaicheng Wang, Xiaoyue Wang, Prudhviraj Naidu, Leon Bergen, Ramamohan Paturi

### Searching for Optimal Per-Coordinate Step-sizes with Multidimensional Backtracking

**Authors:** Frederik Kunstner, Victor Sanches Portella, Mark Schmidt, Nicholas Harvey

### Secure Out-of-Distribution Task Generalization with Energy-Based Models

**Authors:** Shengzhuang Chen, Long-Kai Huang, Jonathan Richard Schwarz, Yilun Du, Ying Wei

### Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images

**Authors:** Zeyu Lu, Di Huang, LEI BAI, Jingjing Qu, Chengyue Wu, Xihui Liu, Wanli Ouyang

### Self-Supervised Reinforcement Learning that Transfers using Random Features

**Authors:** Boyuan Chen, Chuning Zhu, Pulkit Agrawal, Kaiqing Zhang, Abhishek Gupta

### Self-Weighted Contrastive Learning among Multiple Views for Mitigating Representation Degeneration

**Authors:** Jie Xu, Shuo Chen, Yazhou Ren, Xiaoshuang Shi, Hengtao Shen, Gang Niu, Xiaofeng Zhu

### Self-supervised video pretraining yields robust and more human-aligned visual representations

**Authors:** Nikhil Parthasarathy, S. M. Ali Eslami, Joao Carreira, Olivier Henaff

### Semantic HELM: A Human-Readable Memory for Reinforcement Learning

**Authors:** Fabian Paischer, Thomas Adler, Markus Hofmarcher, Sepp Hochreiter

### Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples

**Authors:** Shaokui Wei, Mingda Zhang, Hongyuan Zha, Baoyuan Wu

### Sharp Calibrated Gaussian Processes

**Authors:** Alexandre Capone, Sandra Hirche, Geoff Pleiss

### SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots

**Authors:** Jong Wook Bae, Jungho Kim, Junyong Yun, Changwon Kang, Jeongseon Choi, Chanhyeok Kim, Junho Lee, Jungwook Choi, Jun Won Choi

### [Oral] Siamese Masked Autoencoders

**Authors:** Agrim Gupta, Jiajun Wu, Jia Deng, Fei-Fei Li

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6C

### Single-Call Stochastic Extragradient Methods for Structured Non-monotone Variational Inequalities: Improved Analysis under Weaker Conditions

**Authors:** Sayantan Choudhury, Eduard Gorbunov, Nicolas Loizou

### Small Total-Cost Constraints in Contextual Bandits with Knapsacks, with Application to Fairness

**Authors:** Evgenii Chzhen, Christophe Giraud, Zhen LI, Gilles Stoltz

### Small batch deep reinforcement learning

**Authors:** Johan Obando Ceron, Marc Bellemare, Pablo Samuel Castro

### SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds

**Authors:** Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, Jian Ren

### [Spotlight] Sounding Bodies: Modeling 3D Spatial Sound of Humans Using Body Pose and Audio

**Authors:** Xudong XU, Dejan Markovic, Jacob Sandakly, Todd Keebler, Steven Krenn, Alexander Richard

### Sparse Graph Learning from Spatiotemporal Time Series

**Authors:** Andrea Cini, Daniele Zambon, Cesare Alippi

### SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks

**Authors:** Rainer Engelken

### Sparsity-Preserving Differentially Private Training of Large Embedding Models

**Authors:** Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, Chiyuan Zhang

### [Oral] Spatial-frequency channels, shape bias, and adversarial robustness

**Authors:** Ajay Subramanian, Elena Sizikova, Najib Majaj, Denis Pelli

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6C

### Stability of Random Forests and Coverage of Random-Forest Prediction Intervals

**Authors:** Yan Wang, Huaiqing Wu, Dan Nettleton

### [Spotlight] Stable Diffusion is Unstable

**Authors:** Chengbin Du, Yanxi Li, Zhongwei Qiu, Chang Xu

### Stable and low-precision training for large-scale vision-language models

**Authors:** Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, Ludwig Schmidt

### StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners

**Authors:** Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan

### Star-Shaped Denoising Diffusion Probabilistic Models

**Authors:** Andrey Okhotin, Dmitry Molchanov, Arkhipkin Vladimir, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov, Dmitry Vetrov

### Static and Sequential Malicious Attacks in the Context of Selective Forgetting

**Authors:** Chenxu Zhao, Wei Qian, Rex Ying, Mengdi Huai

### Statistical Limits of Adaptive Linear Models: Low-Dimensional Estimation and Inference

**Authors:** Licong Lin, Mufang Ying, Suvrojit Ghosh, Koulik Khamaru, Cun-Hui Zhang

### [Spotlight] Stein $\Pi$-Importance Sampling

**Authors:** Congye Wang, Ye Chen, Heishiro Kanagawa, Chris Oates

### Stochastic Distributed Optimization under Average Second-order Similarity: Algorithms and Analysis

**Authors:** Dachao Lin, Yuze Han, Haishan Ye, Zhihua Zhang

### Structured Voronoi Sampling

**Authors:** Afra Amini, Li Du, Ryan Cotterell

### SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality

**Authors:** Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna

### Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning

**Authors:** Fuqi Jia, Yuhang Dong, Minghao Liu, Pei Huang, Feifei Ma, Jian Zhang

### [Spotlight] Survival Instinct in Offline Reinforcement Learning

**Authors:** Anqi Li, Dipendra Misra, Andrey Kolobov, Ching-An Cheng

### TIES-Merging: Resolving Interference When Merging Models

**Authors:** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal

### TOA: Task-oriented Active VQA

**Authors:** xiaoying xing, Mingfu Liang, Ying Wu

### Tailoring Self-Attention for Graph via Rooted Subtrees

**Authors:** Siyuan Huang, Yunchong Song, Jiayue Zhou, Zhouhan Lin

### Tartarus: A Benchmarking Platform for Realistic And Practical Inverse Molecular Design

**Authors:** AkshatKumar Nigam, Robert Pollice, Gary Tom, Kjell Jorner, John Willes, Luca Thiede, Anshul Kundaje, Alan Aspuru-Guzik

### [Oral] Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

**Authors:** Guillermo Ortiz-Jimenez, Alessandro Favero, Pascal Frossard

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6A

### Temporal Causal Mediation through a Point Process: Direct and Indirect Effects of Healthcare Interventions

**Authors:** Çağlar Hızlı, ST John, Anne Juuti, Tuure Saarinen, Kirsi Pietiläinen, Pekka Marttinen

### Temporal Continual Learning with Prior Compensation for Human Motion Prediction

**Authors:** Jianwei Tang, Jiangxin Sun, Xiaotong Lin, lifang zhang, Wei-Shi Zheng, Jian-Fang Hu

### [Oral] Tester-Learners for Halfspaces: Universal Algorithms

**Authors:** Aravind Gollakota, Adam Klivans, Konstantinos Stavropoulos, Arsen Vasilyan

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6D

### Text Alignment Is An Efficient Unified Model for Massive NLP Tasks

**Authors:** Yuheng Zha, Yichi Yang, Ruichen Li, Zhiting Hu

### The Bayesian Stability Zoo

**Authors:** Shay Moran, Hilla Schefler, Jonathan Shafer

### [Oral] The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks

**Authors:** Ziqian Zhong, Ziming Liu, Max Tegmark, Jacob Andreas

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6A

### The Drunkard’s Odometry: Estimating Camera Motion in Deforming Scenes

**Authors:** David Recasens Lafuente, Martin R. Oswald, Marc Pollefeys, Javier Civera

### [Spotlight] The Geometry of Neural Nets' Parameter Spaces Under Reparametrization

**Authors:** Agustinus Kristiadi, Felix Dangel, Philipp Hennig

### The Graph Pencil Method: Mapping Subgraph Densities to Stochastic Block Models

**Authors:** Lee Gunderson, Gecia Bravo-Hermsdorff, Peter Orbanz

### [Spotlight] The Pursuit of Human Labeling: A New Perspective on Unsupervised Learning

**Authors:** Artyom Gadetsky, Maria Brbic

### The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data Only

**Authors:** Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay

### The Separation Capacity of Random Neural Networks

**Authors:** Sjoerd Dirksen, Martin Genzel, Laurent Jacques, Alexander Stollenwerk

### The Simplicity Bias in Multi-Task RNNs: Shared Attractors, Reuse of Dynamics, and Geometric Representation

**Authors:** Elia Turner, Omri Barak

### [Oral] The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation

**Authors:** Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, David Fleet

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6C

### The ToMCAT Dataset

**Authors:** Adarsh Pyarelal, Eric Duong, Caleb Shibu, Paulo Soares, Savannah Boyd, Payal Khosla, Valeria A. Pfeifer, Diheng Zhang, Eric Andrews, Rick Champlin, Vincent Raymond, Meghavarshini Krishnaswamy, Clayton Morrison, Emily Butler, Kobus Barnard

### The Tunnel Effect: Building Data Representations in Deep Neural Networks

**Authors:** Wojciech Masarczyk, Mateusz Ostaszewski, Ehsan Imani, Razvan Pascanu, Piotr Miłoś, Tomasz Trzcinski

### The Waymo Open Sim Agents Challenge

**Authors:** Nico Montali, John Lambert, Paul Mougin, Alex Kuefler, Nicholas Rhinehart, Michelle Li, Cole Gulino, Tristan Emrich, Zoey Yang, Shimon Whiteson, Brandyn White, Dragomir Anguelov

### The expressive power of pooling in Graph Neural Networks

**Authors:** Filippo Maria Bianchi, Veronica Lachi

### Thin and deep Gaussian processes

**Authors:** Daniel Augusto de Souza, Alexander Nikitin, ST John, Magnus Ross, Mauricio A Álvarez, Marc Deisenroth, João Paulo Gomes, Diego Mesquita, César Lincoln Mattos

### Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance

**Authors:** Lisha Chen, Heshan Fernando, Yiming Ying, Tianyi Chen

### Time-Reversed Dissipation Induces Duality Between Minimizing Gradient Norm and Function Value

**Authors:** Jaeyeon Kim, Asuman Ozdaglar, Chanwoo Park, Ernest Ryu

### Toolbox for Multimodal Learn (scikit-multimodallearn)

**Authors:** Dominique Benielli, Baptiste Bauvin, Sokol Koço, Riikka Huusari, Cécile Capponi, Hachem Kadri, François Laviolette

### Towards Better Dynamic Graph Learning: New Architecture and Unified Library

**Authors:** Le Yu, Leilei Sun, Bowen Du, Weifeng Lv

### Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?

**Authors:** Hoang Pham, The Anh Ta, Shiwei Liu, Lichuan Xiang, Dung Le, Hongkai Wen, Long Tran-Thanh

### Towards Federated Foundation Models: Scalable Dataset Pipelines for Group-Structured Learning

**Authors:** Zachary Charles, Nicole Mitchell, Krishna Pillutla, Michael Reneer, Zachary Garrett

### Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network

**Authors:** Fuyuan Lyu, Xing Tang, Dugang Liu, Chen Ma, Weihong Luo, Liang Chen, xiuqiang He, Xue (Steve) Liu

### Towards Personalized Federated Learning via Heterogeneous Model Reassembly

**Authors:** Jiaqi Wang, Xingyi Yang, Suhan Cui, Liwei Che, Lingjuan Lyu, Dongkuan (DK) Xu, Fenglong Ma

### Towards a Comprehensive Benchmark for High-Level Synthesis Targeted to FPGAs

**Authors:** Yunsheng Bai, Atefeh Sohrabizadeh, Zongyue Qin, Ziniu Hu, Yizhou Sun, Jason Cong

### [Spotlight] Tracr: Compiled Transformers as a Laboratory for Interpretability

**Authors:** David Lindner, Janos Kramar, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, Vladimir Mikulik

### Training Your Image Restoration Network Better with  Random Weight Network as Optimization Function

**Authors:** man zhou, Naishan Zheng, Yuan Xu, Chun-Le Guo, Chongyi Li

### [Spotlight] Trans-Dimensional Generative Modeling via Jump Diffusion Models

**Authors:** Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Thomas Rainforth, Arnaud Doucet

### TransHP: Image Classification with Hierarchical Prompting

**Authors:** Wenhao Wang, Yifan Sun, Wei Li, Yi Yang

### Transformer as a hippocampal memory consolidation model based on NMDAR-inspired nonlinearity

**Authors:** Dong Kyum Kim, Jea Kwon, Meeyoung Cha, C. Lee

### Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships

**Authors:** ABHRA CHAUDHURI, Massimiliano Mancini, Zeynep Akata, Anjan Dutta

### UDC-SIT: A Real-World Dataset for Under-Display Cameras

**Authors:** Kyusu Ahn, Byeonghyun Ko, HyunGyu Lee, Chanwoo Park, Jaejin Lee

### URL: A Representation Learning Benchmark for Transferable Uncertainty Estimates

**Authors:** Michael Kirchhof, Bálint Mucsányi, Seong Joon Oh, Dr. Enkelejda Kasneci

### Uncertainty Estimation for Safety-critical Scene Segmentation via Fine-grained Reward Maximization

**Authors:** Hongzheng Yang, Cheng Chen, Yueyao CHEN, Scheppach, Hon Chi Yip, DOU QI

### [Spotlight] Uncertainty Quantification over Graph with Conformalized Graph Neural Networks

**Authors:** Kexin Huang, Ying Jin, Emmanuel Candes, Jure Leskovec

### Uncovering Meanings of Embeddings via Partial Orthogonality

**Authors:** Yibo Jiang, Bryon Aragam, Victor Veitch

### Uncovering Neural Scaling Laws in Molecular Representation Learning

**Authors:** Dingshuo Chen, Yanqiao Zhu, Jieyu Zhang, Yuanqi Du, Zhixun Li, Qiang Liu, Shu Wu, Liang Wang

### Understanding Deep Gradient Leakage via Inversion Influence Functions

**Authors:** Haobo Zhang, Junyuan Hong, Yuyang Deng, Mehrdad Mahdavi, Jiayu Zhou

### Understanding How Consistency Works in Federated Learning via Stage-wise Relaxed Initialization

**Authors:** Yan Sun, Li Shen, Dacheng Tao

### Understanding Social Reasoning in Language Models with Language Models

**Authors:** Kanishk Gandhi, Jan-Philipp Fraenken, Tobias Gerstenberg, Noah Goodman

### Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning

**Authors:** Hongyu Zang, Xin Li, Leiji Zhang, Yang Liu, Baigui Sun, Riashat Islam, Remi Tachet des Combes, Romain Laroche

### Understanding the detrimental class-level effects of data augmentation

**Authors:** Polina Kirichenko, Mark Ibrahim, Randall Balestriero, Diane Bouchacourt, Shanmukha Ramakrishna Vedantam, Hamed Firooz, Andrew Wilson

### UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild

**Authors:** Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, Ran Xu

### Universality and Limitations of Prompt Tuning

**Authors:** Yihan Wang, Jatin Chauhan, Wei Wang, Cho-Jui Hsieh

### Unleashing the Full Potential of Product Quantization for Large-Scale Image Retrieval

**Authors:** Yu Liang, Shiliang Zhang, Li Ken Li, Xiaoyu Wang

### [Spotlight] Unpaired Multi-Domain Causal Representation Learning

**Authors:** Nils Sturma, Chandler Squires, Mathias Drton, Caroline Uhler

### Unsupervised Semantic Correspondence Using Stable Diffusion

**Authors:** Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi

### VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset

**Authors:** Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, Jing Liu

### Variational Gibbs Inference for Statistical Model Estimation from Incomplete Data

**Authors:** Vaidotas Simkus, Benjamin Rhodes, Michael Gutmann

### VidChapters-7M: Video Chapters at Scale

**Authors:** Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid

### VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models

**Authors:** Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho

### VisAlign: Dataset for Measuring the Alignment between AI and Humans in Visual Perception

**Authors:** Jiyoung Lee, Seungho Kim, Seunghyun Won, Joonseok Lee, Marzyeh Ghassemi, James Thorne, Jaeseok Choi, O-Kil Kwon, Edward Choi

### VisoGender:  A dataset for benchmarking gender bias in image-text pronoun resolution

**Authors:** Siobhan Mackenzie Hall, Fernanda Gonçalves Abrantes, Hanwen Zhu, Grace Sodunke, Aleksandar Shtedritski, Hannah Rose Kirk

### Visual Programming for Step-by-Step Text-to-Image Generation and Evaluation

**Authors:** Jaemin Cho, Abhay Zala, Mohit Bansal

### WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes

**Authors:** Satoshi Tsutsui, Winnie Pang, Bihan Wen

### [Spotlight] WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting

**Authors:** Yuxin Jia, Youfang Lin, Xinyan Hao, Yan Lin, Shengnan Guo, Huaiyu Wan

### Weakly-Supervised Audio-Visual Segmentation

**Authors:** Shentong Mo, Bhiksha Raj

### What Can We Learn from Unlearnable Datasets?

**Authors:** Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein

### What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks

**Authors:** Taicheng Guo, kehan Guo, Bozhao Nan, Zhenwen Liang, Zhichun Guo, Nitesh Chawla, Olaf Wiest, Xiangliang Zhang

### What can a Single Attention Layer Learn? A Study Through the Random Features Lens

**Authors:** Hengyu Fu, Tianyu Guo, Yu Bai, Song Mei

### What is Flagged in Uncertainty Quantification?  Latent Density Models for Uncertainty Categorization

**Authors:** Hao Sun, Boris van Breugel, Jonathan Crabbé, Nabeel Seedat, Mihaela van der Schaar

### When Do Neural Nets Outperform Boosted Trees on Tabular Data?

**Authors:** Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Ganesh Ramakrishnan, Micah Goldblum, Colin White

### [Oral] When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment

**Authors:** Tianwei Ni, Michel Ma, Benjamin Eysenbach, Pierre-Luc Bacon

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6B

### WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

**Authors:** Sebastian Gerard, Yu Zhao, Josephine Sullivan

### Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization

**Authors:** Nathan Grinsztajn, Daniel Furelos-Blanco, Shikha Surana, Clément Bonnet, Tom Barrett

### [Spotlight] Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis

**Authors:** Alexander Meulemans, Simon Schug, Seijin Kobayashi, nathaniel daw, Gregory Wayne

### XES3G5M: A Knowledge Tracing Benchmark Dataset with Auxiliary Information

**Authors:** Zitao Liu, Qiongqiong Liu, Teng Guo, Jiahao Chen, Shuyan Huang, Xiangyu Zhao, Jiliang Tang, Weiqi Luo, Jian Weng

### YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus

**Authors:** Dave Uthus, Garrett Tanzer, Manfred Georg

### YouTubePD: A Multimodal Benchmark for Parkinson’s Disease Analysis

**Authors:** Andy Zhou, Samuel Li, Pranav Sriram, Xiang Li, Jiahua Dong, Ansh Sharma, Yuanyi Zhong, Shirui Luo, Volodymyr Kindratenko, George Heintz, Christopher Zallek, Yu-Xiong Wang

### [Re] $\mathcal{G}$-Mixup: Graph Data Augmentation for Graph Classification

**Authors:** Ermin Omeragic, Vuk Đuranović

### [Re] Bandit Theory and Thompson Sampling-guided Directed Evolution for Sequence Optimization

**Authors:** Luka Žontar

### [Re] CrossWalk: Fairness-enhanced Node Representation Learning

**Authors:** Luca Pantea, Andrei-Eusebiu Blahovici

### [Re] End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking

**Authors:** Sean McLeish, Long Tran-Thanh

### [Re] Exploring the Role of Grammar and Word Choice in Bias Toward African American English (AAE) in Hate Speech Classification

**Authors:** Priyanka Bose, Chandra shekhar Pandey, Fraida Fund

### [Re] FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

**Authors:** Kyosuke Morita

### [Re] Fairness Guarantees under Demographic Shift

**Authors:** Valentin Buchner, Philip Schutte, Yassin Ben Allal, Hamed Ahadi

### [Re] Numerical influence of ReLU'(0) on backpropagation

**Authors:** Tommaso Martorella, Daniel Garcia

### [Re] On Explainability of Graph Neural Networks via Subgraph Explorations

**Authors:** Yannik Mahlau, Lukas Berg, Leonie Kayser

### [Re] On the Reproducibility of CartoonX

**Authors:** Robin Sasse, Aniek Eijpe, Jona Ruthardt, Elias Dubbeldam

### [Re] On the Reproducibility of “FairCal: Fairness Calibration for Face Verification”

**Authors:** Marga Don, Satchit Chatterji, Milena Kapralova, Ryan Amaudruz

### [Re] Pure Noise to the Rescue of Insufficient Data

**Authors:** Ryan Lee, Seungmin Lee

### [Re] Variational Neural Cellular Automata

**Authors:** Albert Sund Aillet, Simon Sondén

### trajdata: A Unified Interface to Multiple Human Trajectory Datasets

**Authors:** Boris Ivanovic, Guanyu Song, Igor Gilitschenski, Marco Pavone

</details>

<details><summary><h3 style='display: inline;'> Posters Not Being Presented</h3></summary>

### Active Negative Loss Functions for Learning with Noisy Labels

**Authors:** Xichen Ye, Xiaoqiang Li, songmin dai, Tong Liu, Yan Sun, Weiqin Tong

### BCDiff: Bidirectional Consistent Diffusion for Instantaneous Trajectory Prediction

**Authors:** Rongqing Li, Changsheng Li, Dongchun Ren, Guangyi Chen, Ye Yuan, Guoren Wang

### Boosting Verification of Deep Reinforcement Learning via Piece-Wise Linear Decision Neural Networks

**Authors:** Jiaxu Tian, Dapeng Zhi, Si Liu, Peixin Wang, Cheng Chen, Min Zhang

### ConDaFormer: Disassembled Transformer with Local Structure Enhancement for 3D Point Cloud Understanding

**Authors:** Lunhao Duan, Shanshan Zhao, Nan Xue, Mingming Gong, Gui-Song Xia, Dacheng Tao

### Cross-modal Active Complementary Learning with Self-refining Correspondence

**Authors:** Yang Qin, Yuan Sun, Dezhong Peng, Joey Tianyi Zhou, Xi Peng, Peng Hu

### Distributionally Robust Skeleton Learning of Discrete Bayesian Networks

**Authors:** Yeshu Li, Brian Ziebart

### EICIL: Joint Excitatory Inhibitory Cycle Iteration Learning for Deep Spiking Neural Networks

**Authors:** Zihang Shao, Xuanye Fang, Yaxin Li, Chaoran Feng, Jiangrong Shen, Qi Xu

### Fast Model DeBias with Machine Unlearning

**Authors:** Ruizhe Chen, Jianfei Yang, Huimin Xiong, Jianhong Bai, Tianxiang Hu, Jin Hao, YANG FENG, Joey Tianyi Zhou, Jian Wu, Zuozhu Liu

### Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer

**Authors:** Zikai Xiao, Zihan Chen, Songshang Liu, Hualiang Wang, YANG FENG, Jin Hao, Joey Tianyi Zhou, Jian Wu, Howard Yang, Zuozhu Liu

### GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection

**Authors:** Jinggang Chen, Junjie Li, Xiaoyang Qu, Jianzong Wang, Jiguang Wan, Jing Xiao

### Greatness in Simplicity: Unified Self-Cycle Consistency for Parser-Free Virtual Try-On

**Authors:** Chenghu Du, junyin Wang, Shuqing Liu, Shengwu Xiong

### Learning Better with Less: Effective Augmentation for Sample-Efficient Visual Reinforcement Learning

**Authors:** Guozheng Ma, Linrui Zhang, Haoyu Wang, Lu Li, Zilin Wang, Zhen Wang, Li Shen, Xueqian Wang, Dacheng Tao

### Learning Invariant Representations of Graph Neural Networks via Cluster Generalization

**Authors:** Donglin Xia, Xiao Wang, Nian Liu, Chuan Shi

### Masked Two-channel Decoupling Framework for Incomplete Multi-view Weak Multi-label Learning

**Authors:** Chengliang Liu, Jie Wen, Yabo Liu, Chao Huang, Zhihao Wu, Xiaoling Luo, Yong Xu

### MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues

**Authors:** Jia Jinrang, Zhenjia Li, Yifeng Shi

### [Spotlight] Optimizing Prompts for Text-to-Image Generation

**Authors:** Yaru Hao, Zewen Chi, Li Dong, Furu Wei

### [Spotlight] Physics-Driven ML-Based Modelling for Correcting Inverse Estimation

**Authors:** ruiyuan kang, Tingting Mu, Panagiotis Liatsis, Dimitrios Kyritsis

### Preconditioning Matters: Fast Global Convergence of Non-convex Matrix Factorization via Scaled Gradient Descent

**Authors:** Xixi Jia, Hailin Wang, Jiangjun Peng, Xiangchu Feng, Deyu Meng

### RangePerception: Taming LiDAR Range View for Efficient and Accurate 3D Object Detection

**Authors:** Yeqi BAI, Ben Fei, Youquan Liu, Tao MA, Yuenan Hou, Botian Shi, Yikang LI

### Recovering from Out-of-sample States via Inverse Dynamics in Offline Reinforcement Learning

**Authors:** Ke Jiang, Jia-Yu Yao, Xiaoyang Tan

### Spectral Co-Distillation for Personalized Federated Learning

**Authors:** Zihan Chen, Howard Yang, Tony Quek, Kai Fong Ernest Chong

### TexQ: Zero-shot Network Quantization with Texture Feature Distribution Calibration

**Authors:** Xinrui Chen, Yizhi Wang, Renao YAN, Yiqing Liu, Tian Guan, Yonghong He

### The noise level in linear regression with dependent data

**Authors:** Ingvar Ziemann, Stephen Tu, George J. Pappas, Nikolai Matni

### Towards Combinatorial Generalization for Catalysts: A Kohn-Sham Charge-Density Approach

**Authors:** Phillip Pope, David Jacobs

### [Spotlight] Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks

**Authors:** Aoxiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang

</details>

