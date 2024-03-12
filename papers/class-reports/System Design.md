# An EEmaGe System Design: A report for Graduation Project II

$$
\mathbf{Wonjun\ Park,\ Esan\ Woo,\ Hyunwoo\ Lee} \\
\mathrm{Computer\ Science\ and\ Engineering} \\
\mathrm{Konkuk\ University} \\
\mathrm{Seoul,\ South\ Korea} \\
\mathrm{\{kuwjjgjk, esan23, l2hyunwoo\}@konkuk.ac.kr}
$$

### *Abstract*

Admittedly, System Design is so critical that most developers and engineers spend time designing their systems. Nevertheless, everything has changed as time passes. The capacity to change like Agile proved that adaptation is required in some perspectives. For a new semester begins, the report reviews EEmaGe, a framework to reconstruct image and audio from EEG signals, regarding both the design of our system devised in the last semester and the current methodology, comparing if the difference between them existed. We expect that the report demonstrates that the path we took was correct.

$$ \mathbf{Acronym / Abbreviation} \\
\begin{array}{|c|c|}
\hline
\text{Electroencephalography (EEG)} & \text{Brain-Computer Interface (BCI)} \\
\hline
\text{Self-Supervised Learning (SSL)} & \text{Mean Squared Error (MSE)} \\
\hline
\text{Frechet Inception Distance (FID)} & \text{Inception Score (IS)} \\
\hline
\end{array} $$

## I. Introduction

Objects always exist regardless of someone's perception. This influenced that the definitions of looking, seeing, and watching are different. Looking is to toward eyes somewhere, seeing is to perceive things what eyes direct, and watching is to spend time and pay attention to the things [12]. In other words, 'looking' belongs to the 'seeing' set and 'seeing' belongs to the 'watching' set. The visual system of humans performs looking, meaning that, supervision is not required to imitate the system.

BCI, firstly proposed by Vidal [13], has seeked to the key of the human brain where the area has yet been conquered. Disabled people are expected to be benefited to live real lives with non-disabled people, if BCI researches continuously evolve. Among the methodlogies of BCIs, EEG analysis has especially been drawn attention due to its advantages, non-invasive and cost-effectie sensors which are utilized during brain measurements. The analysis, which uses a signal recorded electrical activities of brains [10], is pervasively adopted in medical and research areas to diagnose brain diseases. Even though its effectiveness in those areas, EEG required manual analysis of experts like physicians and researchers [14].

인공지능, 그 속에서도 기계학습은 크게 Supervised Learning, Self-Supervised Learning, Semi-Supervised Learning, Reinforcement Learning의 4가지로 분류합니다. 지금까지의 연구 대부분은 Image와 해당 Image를 바라보는 피험자에게서 관측한 EEG Signal을 Input과 Label(정답)로 연결하여, 주어진 EEG Signal이 정답 Image와 유사한 Image를 재현해내도록 학습하는 Supervised-Learning 방식으로 진행했습니다.

아직까지 연속성이 있는 시계열 Data를 분석하는 데에 한계가 존재하며, 비선형적이기까지한 EEG Signal은 Supervised Learning을 통해 학습하는데에 어려움이 있습니다. 이를 타파하기 위해 Self-Supervised Learning을 접목한 연구가 진행되어 괄목할만한 성과를 이뤘지만, 그럼에도 불구하고 인코딩 방법에 Supervised Learning을 사용했다는 한계가 있습니다.

We propose an EEmaGe which uses SSL autoencoders entailing downstream tasks. EEmaGe is designed to prove our following hypotheses:
* Supervision is not required to construct the human's visual system.
* Excluding visual cues from extracting EEG features are ultimately required.

## II. System Review

In this section, the report addresses the design of the system devised in the last semester along with each task. **A. Image Reconstruction** was regarding the image regeneration from EEG signals. **B. Audio Reconstruction** was a task that restores audio stimuli from recorded EEG signals. In the part of **C. Reconstruction With General BCI**, a unified model was proposed to make original images and sound from EEG signals.

#### A. Image Reconstruction

![Fig 4](./Figure_Image%20Reconstruction%20Framework.png)$\text{Fig 4. A Basic Framework of Image Reconstruction}$

Hinged on our hypotheses, SSL, especially an autoencoder architecture, was adopted to design the system. An autoencoder took an input a pair of eeg and image.

![Fig 5](./Figure_Image%20Reconstruction.JPG)$\text{Fig 5. A Model Architecture for Image Reconstruction}$

#### B. Audio Reconstruction

During the last semester, the team received a feedback to join the EEG-Audio Reconstruction challenge in ICASSP 2024 [1] from the advisor. ICASSP is an acronym of International Conference on Acoustic, Speech, and Signal Processing where the impact score is 3.59 [2]. The challenge was held by KU Leuven, Belgium until December 28th, 2023. Two tasks were presented to the challenge participants: **1) match-mismatch** which is a classification to find a matched EEG-audio pair when five speech segments and an EEG segment are given and **2) regression** which reconstructs mel-spectrogram from EEG signals. The participants were provided with a dataset called SparrKULee [3]. This experience led us to pursue the Generalized BCI which performs in multiple domains such as image and audio. Details of the BCI is tackled in **C. Reconstruction With General BCI for Image and Audio**.

In order to push boundaries of the challenge, we adopted a hypothesis, "Addressing both image and audio data simultaneously is allowed due to their similiarity" [4, 5]. EEG-Image encoders, proposed by other researchers, were utilized to extract meaningful vectors to conduct the task 1 and 2. An EEG Encoder utilized EEG-ChannelNet proposed by Palazzo, et al. [6] and audio encoders are made use of Hubert [7]. The model utilized in this part is on $\text{Fig 1.}$, presented the design to specifically solve the task 1.

![Fig 1](./Figure_Audio%20Reconstruction.png)$\text{Fig 1. A Model Architecture for Audio Reconstruction}$

Each encoders calculated a feature vector, and the vector was compared by its similarity as Cosine Similarity. For the task 2, Pearson Correlation was used as a loss function. We participated the team name as HyperModalityKU in the challenge. $\text{Fig 3.}$ shows the result of the challenge.

![Fig 3-1](./Figure_leaderboard-task1.png)
![Fig 3-2](./Figure_leaderboard-task2.png)$\text{Fig 3. The Leaderboard of The Challenge}$

#### C. Reconstruction With General BCI for Image and Audio

With these approaches, **A. Image Reconstruction** and **B. Audio Reconstruction**, we designed the general BCI which is afforded multimodals, both image and audio.

![Fig 2](./Figure_General%20BCI.png)$\text{Fig 2. A Model Architecture for General BCI}$

As presenting on the $\text{Fig 2}$, the two encoder models, shared their weights while training, were designed.

## III. Methodology

The paper reviewed the system design that planned in the last semester. The project had three goals, image, audio and both. However, the overall result of the EEG-audio challenge revealed that the direct achievement of the general BCI has obstacles so far. In other words, in order to achieve the general BCI, each goal which means image and audio reconstruction should be preceded. This section shows a framework called EEmaGe to reconstruct visual stimuli from EEG signals.

#### A. EEmaGe

Focusing back on the image reconstruction, SSL and its downstream task has still adopted for the current model to achieve the exclusion of the supervision. EEmaGe is an autoencoder-based model architecture which gets an input $(e, i)$ pair where $e$ is an EEG and $i$ is an image.

**Training** Two autoencoders which their encoders share weights with themselves comprise the architecture. Each encoder has a preprocessing block to feed-forward into the same encoder structure. Those autoencoders are back-propagated at the same time with a loss function MSE $\cdots (1)$. Specifically, the loss function of the compounded model is a sum of a loss from the EEG autoencoder $L_{\text{eeg}}$ and a loss from the image autoencoder $L_\text{image}$. The formula is written in $\cdots (2)$.

$$
(1)\ \text{MSE} = {1 \over n} \sum_{i=1}^n {(Y_i - \hat{Y_i})}^2 \\
(2)\ \text{Loss} = L_{\text{eeg}} + L_{\text{image}}
$$

**Downstream Task** A downstream task is defined as reconstructing images from EEG signals with an autoencoder. Transferring the EEG encoder and the image decoder from EEmage, inferences of the autoencoder implement to generate images. This task can be differentiated with **1)** utilizing the autoencoder as a foundation model itself and **2)** fine-tuning the autoencoder to maximize its performance. Even though **2)** contains the supervision, the novelty of the research is still found on the proposal of the EEG foundational model. We will present the two cases to sound the performance of EEmaGe.

#### B. Performance Evaluation

To evaluate the performance, FID [15], the current standard metric to assess generative models, is planned to use. Unlike IS which evaluates the distribution of the generated images, FID evaluates the distributions from the original images to the generated images.

## IV. Implementation

#### A. EEG-Image Pair Dataset

In the task of visual reconstruction, two qualified available datasets were compared.

**Thoughtviz Dataset** Yet, Thoughtviz [8] dataset, originated from Kumar, et al. [9] which collected an EEG dataset for the speech recognition task, utilized imaginary images of participants. The dataset collected by relying on the thought of the participants has an alpha wave data induced by thinking.

**PerceiveLab Dataset** In fact, the EEG beta waves are dominant while the eyes open [10]. Palazzo, et al [11] collected the pairs of EEG and image data from six participants. The ImageNet subset, consisting of fifty images per class where the number of the class is forty, were selected by those researchers. Consequently, 12,000 EEG sequences (2000 images * 6 participants) were gathered via 128 EEG channels. Few sequences were excluded through preprocessing so that 11,466 were valid to account for the opened dataset.

We designed two experiments that **1)** using the EEG-image pair already matched in the dataset and **2)** shuffling the pair to ultimately achieve our second hypothesis. An input $(e, i)$ pair where $e$ is an EEG and $i$ is an image were given to the model.

#### B. Development Environment

A bare metal computer, equipped with Intel i5-10400F CPU, GTX 1660Ti GPU, and two Samsung 8 GB 2666Hz RAMs, runs for the model training. The versions of Python and its framework are the following: Python 3.10.12, PyTorch 2.2, and Tensorflow 2.13.0. Further details are specified in the requirements.txt file at the project repository.

## V. Conclusion

As a new semester has been started, the report inspects the design of the system called EEmaGe. This framework was formulated to reconstruct human experience from EEG signals. Checking how the system and its compoments were designed concretely, the details of the components have been compounded. The report concludes that our design conducted in the last semester has been compelling to develop. We expect that this approach ultimately contributes to develop a general BCI application.

## References

* [1] https://exporl.github.io/auditory-eeg-challenge-2024/ . accessed in Mar 11, 2024 [URL]
* [2] https://www.resurchify.com/impact/details/110544 . accessed in Mar 11, 2024 [?]
* [3] Accou, Bernd, et al. "SparrKULee: A Speech-evoked Auditory Response Repository of the KU Leuven, containing EEG of 85 participants." bioRxiv (2023): 2023-07. [MLA]
* [4] Likhosherstov, Valerii, et al. "Polyvit: Co-training vision transformers on images, videos and audio." arXiv preprint arXiv:2111.12993 (2021). [MLA]
* [5] Y. Gong, A. H. Liu, A. Rouditchenko and J. Glass, "UAVM: Towards Unifying Audio and Visual Models," in IEEE Signal Processing Letters, vol. 29, pp. 2437-2441, 2022, doi: 10.1109/LSP.2022.3224688. [IEEE]
* [6] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt and M. Shah, "Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 11, pp. 3833-3849, 1 Nov. 2021, doi: 10.1109/TPAMI.2020.2995909. [IEEE]
* [7] W. -N. Hsu, B. Bolte, Y. -H. H. Tsai, K. Lakhotia, R. Salakhutdinov and A. Mohamed, "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451-3460, 2021, doi: 10.1109/TASLP.2021.3122291. [IEEE]
* [8] Tirupattur, Praveen, et al. "Thoughtviz: Visualizing human thoughts using generative adversarial network." Proceedings of the 26th ACM international conference on Multimedia. 2018. [MLA]
* [9] Kumar, Pradeep, et al. "Envisioned speech recognition using EEG sensors." Personal and Ubiquitous Computing 22 (2018): 185-199. [MLA]
* [10] Teplan, Michal. "Fundamentals of EEG measurement." Measurement science review 2.2 (2002): 1-11. [MLA]
* [11] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano and M. Shah, "Generative Adversarial Networks Conditioned by Brain Signals," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 3430-3438, doi: 10.1109/ICCV.2017.369. [IEEE]
* [12] https://www.britannica.com/dictionary/eb/qa/see-look-watch-hear-and-listen , accessed in Mar 4 2024. [URL]
* [13] Vidal, Jacques J. "Toward direct brain-computer communication." Annual review of Biophysics and Bioengineering 2.1 (1973): 157-180. [MLA]
* [14] https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview , accessed in Mar 4 2024. [URL]
* [15] Heusel, Martin, et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium." Advances in neural information processing systems 30 (2017). [MLA]
