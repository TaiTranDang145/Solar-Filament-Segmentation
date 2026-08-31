# TASK

Participants are invited to develop algorithms for the automated segmentation of solar filaments. Any computational approach may be employed, including traditional image-processing techniques, machine learning methods, and state-of-the-art deep neural network architectures.

The objective of this challenge is to develop an algorithm---whether novel or existing---that generates accurate segmentation masks for individual solar filaments. For each filament, the predicted mask should capture the complete extent of the filament, including its fine-scale structures, while minimizing the inclusion of non-filament regions. The challenge emphasizes robust and precise delineation of filament morphology across a diverse set of solar observations.


# GOAL

This competition focuses on the pixel-level precision of filament segmentation masks. The primary evaluation criteria is the Panoptic Quality metric ([10.1109/CVPR.2019.00963](/home/lok/Downloads/1801.00868v1.pdf)). In addition to segmentation accuracy, the evaluation framework incorporates penalties related to fragmentation and over-merging (e.g., one-to-many and many-to-one correspondences between predicted and ground-truth filament structures), as well as the end-to-end computational efficiency of the proposed method.



# Announcements

[Aug 12, 2026] Our leaderboard is now re-scored.
[Aug 9, 2026] We added a self-evaluation notebook for participants. Please see the "Self Evaluation" section.
[Aug 7, 2026] We are going to update our evaluation methodology to address the raised issues. The details are explained below. The new scoring metric will be in effect soon; all previous submissions will be updated as well. The exact time the leaderboard is to be updated depends on when Kaggle resets the scoring mechanism.
[Aug 20, 2026] Looking at some submissions, it is clear to us that some participants are spending their time and energy on hacking the system rather tackling the actual problem. We encourage honest participants to continue their work without comparing their scores with others. Any PQ score of greater than 0.35 is of great value to us.

# Important Dates

July 10, 2026: Competition is launched. Contestants can start working on the challenge.
Nov 15, 2026: Deadline for contest teams to submit final reports and solutions.
Nov 30, 2026: The winning teams will be announced.
Dec 14-17, 2026: (Optional) IEEE BigData conference will take place at Phoenix, AZ. The winners of this BigDataCup competition (as well as all other BigDataCups) will be announced

# Final Submission
(Added: Aug 7, 2026)

When ready, prepare the following items and submit them through this google form. The items are:

    A technical report: A 4-page technical report is required. Please use this Overleaf template for your report. More details about the content of this report are provided in the template. The final report should be submitted as a single pdf file.
    Code Repository: A publicly accessible link to a Git-based repository such as GitHub, GitLab, or Bitbucket, is required. The organizing committee should access your source code without requiring additional access permissions or approval. Within this repository, a file named requirements.txt must include all utilized packages and their corresponding versions. Also, a jupyter notebook should illustrate the entire pipeline. The committee should not require any additional file to reproduce the submitted predictions.

All participants who wish to be considered for the final evaluation must complete the Google form. Please ensure that all information provided is accurate and complete. Any errors or omissions that prevent the judging committee from accessing the required materials or contacting the participant may result in the submission being deemed ineligible for final evaluation.

# Description
## What Are Solar Filaments?

Filaments (as shown below) are dense "clouds" of solar material suspended by magnetic field lines above photospheric neutral lines. The significance of filaments for space weather research lies in the fact that they are at the core of solar eruptions, including Coronal Mass Ejections (CMEs), solar flares, and Solar Energetic Particle (SEP) storms. An Earth-directed CME can cause enormous damage to the electric power grid, disrupt GPS systems, create radiation hazards for passengers and crew on polar flights, and be lethal to astronauts traveling outside the protective bubble provided by the Earth's magnetosphere. This is why solar filaments are a critical event type in space weather research and forecasting operations.

![img1.png](img1.png)

## What Is MAGFiLO?

MAGFiLO, short for Manually Annotated GONG Filaments from H-Alpha Observations, is the ground-truth data of filaments' segmentation mask ([10.1038/s41597-024-03876-y](/home/lok/Downloads/s41597-024-03876-y.pdf)). It is created for training and evaluation of machine-learning algorithms, and for evaluation of traditional computer-vision algorithms.

The illustration below shows an H-Alpha observation of the Sun, in which solar filaments have been identified by expert human annotators. Although the dataset includes additional annotations, such as bounding boxes, filament spines, and class labels, this competition focuses exclusively on the segmentation masks of solar filaments.

MAGFiLO is the testbed for all participants' models.

See the Data tab for more details.

![img2.png](img2.png)

## What Are the Main Challenges?

Although recent advances in object segmentation have led to remarkable performance across many domains, solar filament segmentation remains a challenging task for several reasons:

    Fine-scale structures. Accurately capturing fine filament structures, such as barbs, remains difficult. Barbs are thin, thread-like features that extend from the main body of a filament along a characteristic orientation. Their orientation relative to the filament spine carries important information about the filament's underlying magnetic field configuration.

    Background noise and image quality. Distinguishing filament material (dark regions) from background structures and noise is nontrivial. Since the observations used in this competition originate from ground-based observatories, they are affected by various sources of noise and imaging artifacts. Accurately identifying small-scale filament structures while suppressing background noise remains a significant challenge.

    Structural continuity. Existing segmentation algorithms often struggle to identify solar filaments as contiguous physical structures. Instead, they may produce fragmented segmentations or segment clusters of nearby dark regions ("islands"), resulting in incomplete or physically inconsistent representations of the filament morphology.


# Evaluation

(Updated: Aug 7, 2026)

Although participants may use other external data sources, please note that: (1) for inference of ML models, only the H-alpha images provided in the test directory will be used; and (2) the models may not use any other ground-truth metadata for training.

All submissions are judged based on the following rubric:

    Quantitative Comparison (70%):
        Panoptic Quality metric (see the Leaderboard Ranking section below for more details.)
        Distribution of Dice scores
        Distribution of IoU scores
        Distribution of one-to-many and many-to-one relations between the ground-truth and predicted segmentations.
    Qualitative Comparison (30%):
        Detailed description of the entire pipeline (from preprocessing to final prediction, including the architecture of the utilized algorithm).
        The apparent morphology of predicted segmentations on H-Alpha images.
        The quality of code (modularity and documentation)

Note: The evaluation of the above criteria is contingent upon the availability of the source code. Please refer to the Open-Access Policy section below.

# Leaderboard Ranking

(Added: Aug 7, 2026)

The ranking on the leaderboard is one of the considerations in our evaluation (see above rubric.)

The submissions will be evaluated against the ground-truth data (MAGFiLO) using the **Panoptic Quality** metric which is defined as follows:

$$
\operatorname{PQ}(Y, \hat{Y}) =
\frac{\sum_{(y,\hat{y})\in TP} \operatorname{IoU}(y,\hat{y})}
{|TP| + 0.5|FP| + 0.5|FN|}
$$

where

- $Y$ is a set of all ground-truth segments $y$,
- $\hat{Y}$ is a set of all predicted segments $\hat{y}$,
- IoU computes the Intersection-over-Union as shown below,
- TP, FP, and FN denote sets of all true-positive, false-positive, and false-negative cases, respectively, and
- $|\cdot|$ is the set cardinality operation.

Intersection-over-Union is measured as follows:

$$
\operatorname{IoU}(y,\hat{y}) =
\frac{|y \cap \hat{y}|}{|y \cup \hat{y}|}
= \frac{\sum(y \odot \hat{y})}
{\sum(y \oplus \hat{y} \ominus y \odot \hat{y})}
$$

For more information about Panoptic Quality metric, see the original paper ([10.1109/CVPR.2019.00963](/home/lok/Downloads/1801.00868v1.pdf)).

**Note:** There are further verification strategies addressing certain edge cases. Appropriate messages will be shown after a submission is uploaded, when/if such cases are identified.

# Self Evaluation

(Added: Aug 7, 2026)

A self-evaluation notebook containing a few scores and plots is now available. Participants can use this notebook to have a better understanding of the effectiveness of their segmentation algorithms.

    Notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

# Submission File

We expect the participants to upload a single CSV file for the entire test set, where each row corresponds to one predicted filament, as shown below.

```text
filament_id             | segmentation_rle       |
------------------------|------------------------|
20150125172714Mh_1      | "f8uSDds ... VQNC"     |
20150125172714Mh_2      | "KHT%$HD ... 9>km"     |
20150125172714Mh_3      | "YQNEgn1 ... BH6^"     |
...                     | ...                    |
20170501024112Bh_1      | "HBy4d6D ... 97*D"     |
```

- Column `filament_id` contains unique ids for each filament, e.g., `20150125172714Mh_2`.
- Column `segmentation_rle` contains RLE counts, each encoding the mask corresponding to one filament, e.g., ``^Vj02jo16I50201`PNA)olc0N19G1N11O3L0104JYamT3``. Do not include quotations (`'` or `"`).

**Note:** You only need to store RLE Counts; no need to store RLE Size. The Size is fixed for all images: 2048 X 2048 pixels.

To convert your output (e.g., masks, polygons, etc.) to the expected format (RLE Counts), please use the `pycocotools` Python package, specifically, the methods `annToMask`, `decodeMask`, and `encodeMask`. For more details, see `coco.py`.

**Note:** The tail strings in the `filament_id` column (e.g., `_2`) are only to make the rows unique. For example, if your algorithm identifies 3 filaments in a given image with id `20150125172714Mh`, you should generate 3 rows with the following keys: `20150125172714Mh_1`, `20150125172714Mh_2`, and `20150125172714Mh_3`. As long as the tail strings render the rows unique, and the image id remains unchanged, your filament id is acceptable.

**Note:** The number of ground-truth segmentations may also be different from the number of predicted ones. Therefore, the evaluation method matches the predicted and ground-truth segmentations based on their actual overlap, not their index.

The scoreboard will show contestants the mean Dice score for ~50% of the images in the test set. The results on the remaining images will be visible only to the organizers of the competition. For a fair evaluation pipeline, the two scores should be fairly close.
