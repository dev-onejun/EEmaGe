# (Working Title) EEmaGe: EEG-based Image Generation for Visual Reconstruction

**Hypothesis**

1) Supervision is not required to construct the human's visual system.
2) Visual cues are ultimately discouraged from extracting the EEG features.

***Abstract***

**Version 1**
Visual reconstruction from EEG brain signals becomes possible with the advancement of AI. Recent researches showed that, in designed experiments, recorded EEG while presenting images enables ML models to reconstruct the images. Nevertheless, even though breakthroughs in AI have begun with imitating the human system, previous frameworks were not similar with the visual system. This research postulated that supervised learning should be avoided to build a reconstruction framework as well as visual cues ultimately keep away from the training to extract meaningful EEG features. The research proposes a novel framework called EEmaGe with *a self-supervised autoencoder and its downstream* to regenerate the human vision appropriately. *The framework showed the state-of-the-art performance in (cosine singularity) metrics. As the RE2I approach, the research is expected to contribute to solve the secret of the human brain which has not yet been solved.*

**Version 2**
Visual reconstruction from EEG has been paved with the advancement of AI. Recent studies have demonstrated the feasibility of reconstructing images from EEG recordings in designed experiments. Nevertheless, even though breakthroughs in AI have begun with imitating the human system, these frameworks lack resemblance to the visual system of the human. To address this challenge, this research proposes a novel framework called EEmaGe which utilizes self-supervised learning to reconstruct images from raw EEG data. Unlike supervised learning methods, which rely on labeled training data, the framework employs *a self-supervised autoencoder and downstream task* to mimic human vision without visual cues. *The experimental results showcase the state-of-the-art performance of the framework in metrics related to cosine singularity.
As the RE2I approach, the research has the potential to contribute to advance our knowledge of the intricacies of the human brain and to develop more sophisticated AI systems that effectively mock human visual perception.*

**Acronym/Abbreviation**
* electroencephalogram?/electroencephalography? (EEG)
* Artificial Intelligence (AI)
* Machine Learning (ML)
* Reconstruction from EEG to Image (RE2I)
* Convolutional Neural Networks (CNN)
* Small-World Neural Networks (SWNet)

## I. Introduction

**WHY IS SUPERVISION NOT REQUIRED TO CONSTRUCT HUMAN VISUAL SYSTEM?**
It is a fact that things always exist regardless of someone's perception. This influenced that the definitions of looking, seeing, and watching are different. Looking is to toward eyes somewhere, seeing is to perceive things what eyes direct, and watching is to spend time and pay attention to the things [4]. In other words, 'looking' belongs to the 'seeing' set and 'seeing' belongs to the 'watching' set. The visual system of humans performs looking, meaning that, supervision is not required to imitate the system. **IS VISUAL RECONSTRUCTION TASK SAME AS LOOKING(-VISUAL SYSTEM)?**

EEG, pervasively used in medical area to diagnose brain diseases, is signal recorded electrical activities of brains [5]. The recording is allowed via non-invasive and cost-effective sensors. Recently, many researches founded that EEG is able to reproduce visual experiences.

AI has been adnvaced with imitating the system of human beings. For instance, Neural Network [1] mimicked the human nervous system as well as its advancement like CNN [2] and SWNet [3] (imitate/mimic/resemble).

## II. Related Works

## III. Experiment

## IV. Implementation

## V. Conclusion

## References

***Citations are followed MLA format so far (except URL)***

* [1] McCulloch, Warren S., and Walter Pitts. "A logical calculus of the ideas immanent in nervous activity." The bulletin of mathematical biophysics 5 (1943): 115-133.
* [2] Fukushima, Kunihiko. "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position." Biological cybernetics 36.4 (1980): 193-202.
* [3] Javaheripi, Mojan, Bita Darvish Rouhani, and Farinaz Koushanfar. "SWNet: Small-world neural networks and rapid convergence." arXiv preprint arXiv:1904.04862 (2019).
* [4] https://www.britannica.com/dictionary/eb/qa/see-look-watch-hear-and-listen
* [5] Teplan, Michal. "Fundamentals of EEG measurement." Measurement science review 2.2 (2002): 1-11.
