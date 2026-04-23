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
