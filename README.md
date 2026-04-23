# Geometrically based Human Phenotype Classification utilizing Face Meshes with PointNet

## Paper

* [ ]  Threshold Analysis of how many samples are required for a >=0.99 performance -> Not sure if this makes sense as the number of samples most certainly will differ depending on the category and number of points feed into the model
* [ ]  Select best of 5-fold models for LIRICAL analysis (OPTIONAL)
* [X]  "kind of" XAI by visualizing the importance value of the points for each stage
* [X]  Ablation Study
* [X]  HPO/GMDB Dataset description
* [X]  Validate the predictions by experts -> run on external testing set (on 3 annotations)
  * [X]  Testing on syndromes not included in the training/validation set -> is it still possible to predict their related HPO-terms?
  * [X]  Matching the predicted HPO-terms with their actual HPO-terms per syndrome and their frequency (Visualization)
* [X]  Re-run experiment with gmdb data as negative cases or in combination with UTKFaces if not enough gmdb images are available for specific HPO-term
* [X]  Looking into the preprocessing pipeline of medapipe
* [ ]  Threshold on the models performance (webapp)
* [ ]  Threshold on the performance predictions (webapp)
* [X]  Look at the best and worst performing models
* [ ]  Run on UTK & GMDB non-training samples to compute prevelance of each HPO-term
  * [ ]  Issue: limitation of the data for some HPO-terms (e.g., Hypotelorism, Hypertelorism)
  * [ ]  Use ~ >80% accuracy models to compute prevelance
* [ ]  Look at the average classification confidence for each hpo-term

Introduction

* Genetic disorders -> how phenotypic descriptors is sometimes difficult and timeconsuming and subjective task
* Human Phenotype Ontology

Related Work

* Explore other phenotypic descriptor tools (LIRICAL)
* Explore other syndrome classification tools (DeepGestalt, PhenoScore)
* Classification of phyenotpic decsriptors based on genes (Phenomizer [https://bio.tools/phenomizer])

Method

* Dataset
* Metrics
* Model-Architecture
* Training Strategy (Feature elimination, dynamic parameterize model)

Result

* Ablation Study Results
* Best performing model from ablation study

  * experiments on test ("extern validation") set (HPO: Ben, Peter, Shahida)
  * Experiments to badly performing terms
    * Are they not visible?
    * Threshold Analysis of how many samples are required for a >=0.99 performance
  * Experiment on non-visible but great performing terms (likely confounding effect)
  * testing on syndromes not included in the training/validation set
  * LIRICAL experiments?

Discussion

* Results
* Limitations

Conclusion

## Ideas for future improvements

* Experiment with a greater negative sample set (e.g., using full UTKFaces dataset instead of selected samples) and using class weights.

## Experiments / Ablation Study

```
for d in 3 2; do  # Dimensions
  for o in False True; do  # Face Outline
    for l in 0 0.05 0.1; do  # Soft Labels
      for t in 0.01 0.05 0.1; do  # Feature Importance Threshold
        for m in '[]' '["age","gender","ethnicity"]'; do
          sbatch slurm_train.sh -d $d -o $o -l $l -t $t -m $m
        done
      done
    done
  done
done
```

## HPO-Term visibility in Face Mesh

The results (F1-Score and AUROC) are from the experiment with the following configuration:

* Dimensions: 3
* Face Outline: False
* Soft Labels: 0.05
* Feature Importance Threshold: 0.01
* Meta Data: [age, gender, ethnicity]
* Maximum Epochs: 25
* Early Stopping: 5 patience


| Category           | HPO        | Description                      | Visible | F1-Score       | AUROC          | Comment                                                                                                                      |
| ------------------ | ---------- | -------------------------------- | ------- | -------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Mouth              | HP:0000160 | Narrow mouth                     | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000154 | Wide Mouth                       | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0002714 | Downturned Corners of mouth      | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0410030 | Cleft lip                        | No      | 15.38 ± 34.4 | 20.00 ± 19.2  | mediapipe is not detecting cleft lip                                                                                         |
|                    | HP:0000343 | Long Philtrum                    | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000289 | Broad Philtrum                   | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000322 | Short Philtrum                   | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0011829 | Narrow Philtrum                  | Yes     | 75.84 ± 26.8 | 82.98 ± 14.2  | Narrow Philtrum performs worse -> only 170 samples total compared to 460-760 samples in the other philtrum related HPO-Terms |
|                    | HP:0000233 | Thin Vermillion border           | Unsure  | 100.0 ± 0.00 | 100.0 ± 0.00  | The border would be represented by a few points and not clearly visible.                                                     |
|                    | HP:0012471 | Thick Vermillion border          | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | The border would be represented by a few points and not clearly visible.                                                     |
|                    | HP:0000219 | Thin upper lip Vermillion        | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | The border would be represented by a few points and not clearly visible.                                                     |
|                    | HP:0000179 | Thick lower lip Vermillion       | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | Annotated as "Thick upper lip Vermillion" #Problem? (Artifects?)                                                             |
|                    | HP:0010803 | Everted upper lip Vermillion     | Unsure  | 99.70 ± 0.67 | 99.71 ± 0.66 |                                                                                                                              |
|                    | HP:0000232 | Everted lower lip Vermillion     | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000278 | Retrognathia                     | No      | -              | -              | (Skull Morphology) not annotated at all                                                                                      |
|                    | HP:0000347 | Micrognathia                     | No      | -              | -              | (Skull Morphology) not annotated at all                                                                                      |
|                    | HP:0000303 | Mandibular prognathia            | No      | -              | -              | (Skull Morphology) not annotated at all                                                                                      |
| Chin               | HP:0000307 | Pointed Chin                     | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  | probably only in 3D visible                                                                                                  |
| Nose               | HP:0000430 | Underdeveloped nasal alae        | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0009928 | Thick nasal alae                 | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0000463 | Anteverted nares                 | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0009931 | Enlarged naris                   | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0000455 | Broad nasal tip                  | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0012810 | Wide nasal base                  | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0000431 | Wide nasal Bridge                | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000446 | Narrow nasal Bridge              | Yes     | 98.95 ± 2.35  | 99.00 ± 2.24  | Performs worse than HP:0000431 "wide nasal bridge"                                                                           |
|                    | HP:0005280 | Depressed nasal Bridge           | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000437 | Depressed nasal tip              | Unsure  | 100.0 ± 0.00 | 100.0 ± 0.00 | due to an artifact of media pipe the depressed nasal tip might not be detectable as all nose structures are quite flat       |
|                    | HP:0003189 | Long Nose                        | Yes     | 98.68 ± 1.91  | 98.72 ± 1.84  |                                                                                                                              |
|                    | HP:0003196 | Short Nose                       | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0002000 | Short columella                  | Yes     | 100.0 ± 0.00 | 100.0 ± 0.00 | might occure together with "depressed nasal Bridge"                                                                          |
| Face Shape         | HP:0000275 | Narrow face                      | Yes     | 99.62 ± 0.84  | 99.63 ± 0.83  |                                                                                                                              |
|                    | HP:0012368 | Flat Face                        | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000325 | Triangular Face                  | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0004428 | Elfin Facies                     | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000280 | Coarse facial Features           | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0004493 | Craniofacial hyperostosis        | Unsure  | 29.33 ± 40.4  | 56.67 ± 14.9  | works well but is it really visible in the face mesh? -> Probably not because it's the bone structure                        |
| Mid Face           | HP:0011800 | Midface retrusion                | Yes     | 99.65 ± 0.78  | 99.66 ± 0.77  |                                                                                                                              |
|                    | HP:0000293 | Full cheeks                      | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0010669 | Hypoplasia of the zygomatic bone | Unsure  | -              | -              |                                                                                                                              |
| Forehead           | HP:0009890 | High anterior hairline           | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | probably more the relation as the points do not reach the hairline but approximately the midline of the forehead             |
|                    | HP:0000294 | Low anterior hairline            | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | probably more the relation as the points do not reach the hairline but approximately the midline of the forehead             |
|                    | HP:0000290 | Abnormality of the forehead      | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | probably more the relation as the points do not reach the hairline but approximately the midline of the forehead             |
|                    | HP:0000348 | High forehead                    | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | probably more the relation as the points do not reach the hairline but approximately the midline of the forehead             |
|                    | HP:0002007 | Frontal Bossing                  | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  | probably more the relation as the points do not reach the hairline but approximately the midline of the forehead             |
|                    | HP:0000337 | Broad forehead                   | Yes     | 99.48 ± 1.17 | 99.49 ± 1.14  |                                                                                                                              |
|                    | HP:0000341 | Narrow forehead                  | Yes     | 99.69 ± 0.69  | 99.70 ± 0.68  |                                                                                                                              |
| Eyebrow            | HP:0045075 | Sparse eyebrow                   | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000574 | Thick eyebrow                    | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0002223 | Absent eyebrow                   | Unsure  | -              | -              |                                                                                                                              |
|                    | HP:0002553 | Highly arched eyebrow            | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0000664 | Synophrys                        | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
| Periorbital Region | HP:0000336 | Prominent supraorbital ridges    | Yes     | 100.0 ± 0.00  | 100.0 ± 0.00  |                                                                                                                              |
|                    | HP:0011231 | Prominent eyelashes              | No      | 96.92 ± 6.47  | 97.25 ± 5.61  |                                                                                                                              |
|                    | HP:0000637 | Long palpebral fissure           | Yes     | 96.92 ± 2.54  | 96.97 ± 2.52  | data augmentation flip for improvement?                                                                                      |
|                    | HP:0012745 | Short palpebral fissure          | Yes     | 86.44 ± 13.7  | 88.78 ± 10.2  | data augmentation flip for improvement?                                                                                      |
|                    | HP:0000581 | Blepharophimosis                 | Unsure  | 20.40 ± 31.6  | 53.01 ± 4.23  |                                                                                                                              |
|                    | HP:0000286 | Epicanthus                       | Unsure  | 100.0 ± 0.00  | 100.0 ± 0.00 |                                                                                                                              |
|                    | HP:0100539 | Periorbital edema                | No      | -              | -              |                                                                                                                              |
|                    | HP:0000494 | Downslanted palpebral fissures   | Yes     | 99.83 ± 0.37  | 99.84 ± 0.37  |                                                                                                                              |
|                    | HP:0000582 | Upslanted palpebral fissure      | Yes     | 96.57 ± 3.93  | 96.80 ± 3.60  |                                                                                                                              |
| Eye                | HP:0000508 | Ptosis                           | Yes     | 98.53 ± 3.30  | 98.63 ± 3.07  |                                                                                                                              |
|                    | HP:0000486 | Strabismus                       | Yes     | 96.04 ± 5.44  | 96.39 ± 4.76  |                                                                                                                              |
|                    | HP:0000525 | Abnormality Iris morphology      | Yes     | 78.99 ± 12.2  | 76.67 ± 18.6  |                                                                                                                              |
|                    | HP:0000568 | Microphthalmia                   | Yes     | 60.34 ± 37.1  | 62.30 ± 22.1  |                                                                                                                              |
|                    | HP:0000520 | Proptosis                        | Unsure  | 74.19 ± 17.7  | 75.52 ± 18.1  | Testing required!                                                                                                            |
|                    | HP:0000316 | Hypertelorism                    | Yes     | 99.84 ± 0.36  | 99.84 ± 0.36  |                                                                                                                              |
|                    | HP:0000601 | Hypotelorism                     | Yes     | 61.84 ± 34.8  | 73.57 ± 13.8  |                                                                                                                              |
