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

Note: GitHub automatically truncates files larger than 512 KB. To have all papers display on GitHub, we've split the file up by session.

[Poster Session 1](schedule/Poster_Session1.md)

[Poster Session 2](schedule/Poster_Session2.md)

[Poster Session 3](schedule/Poster_Session3.md)

[Poster Session 4](schedule/Poster_Session4.md)

[Poster Session 5](schedule/Poster_Session5.md)

[Poster Session 6](schedule/Poster_Session6.md)

[Posters Not Presented](schedule/Not_Presented_Posters.md)

