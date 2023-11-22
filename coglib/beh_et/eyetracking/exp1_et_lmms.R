library(ordinal)
library(tidyr)
library(dplyr)
library(lmerTest)
library(BayesFactor)
library(bayestestR)
library(emmeans)
library(ggplot2)
library(ggeffects)
library(rlang)
library(afex)

"
The following are the eye-tracking data analyses for experiment 1 of the Cogitate Consortium.
The models below follow pre-registration 4.0: https://osf.io/gm3vd

The model inputs (data tables) are csv files outputted from the eye-tracking data quality-checks,
which were done separately in Python; all codes can be found in the consortium's github.

NOTE: one of the labs (CF) had no pupil data, which means it doesn't have reliable
pupil and blink information. Therefore, these subjects are excluded from pupil and blink
analyses.

In the following, the dependent variable changes, while the tested effects are identical across all models.
As per the pre-registration:

'
We will analyze the following dependent variables: mean  fixation  distance,  mean  blink  rate,  mean  saccade  amplitude  and  mean  pupil  size.
Each variable will be analyzed using a LMM with category (Faces/Objects/Letters/False-fonts) and task (relevant/irrelevant) as fixed effects
and subject and item as random effects.
This will initially be conducted on the first time window, where a stimulus has been presented in all trials (0-0.5s).
Then, to further test whether the dependent variables change as a function of stimulus duration, we will focus only on 1.5s trials,
and run another LMM with category, task, and time window (0-0.5s, 0.5-1.0s, 1.0-1.5s) as fixed effects and subject and item as random effects.
'

@author: RonyHirsch
"


## Data ---------------------

"
All the data needed for the following analyses is in the following df
"


modalities <- c("fMRI", "MEG", "ECoG")  # all three modalities, this is used for per-modality modelling


data <- read.csv("lmm_df_phase3.csv")  # all data
data <- filter(data, has_ET_data != FALSE)  # filter out those who don't have eyetracking data

# categorical fixed effects
data$subCode <- factor(data$subCode)  # subject code
data$modality <- factor(data$modality)  # fMRI, MEG, ECoG
data$stimType <- factor(data$stimType)  # face, object, letter, falseFont
data$isTaskRelevant <- factor(data$isTaskRelevant)  # True, False
data$plndStimulusDur <- factor(data$plndStimulusDur)  # 0.5, 1, 1.5 (seconds)
data$stimCode <- factor(data$stimCode)  # stimulus code (unique per item)
data$timeWindow <- factor(data$timeWindow)  # first window (0-500ms), second window (500-1000ms), third window (1000-1500ms)




## =========================== ANALYSIS 1 ===========================

"
In this analysis, we test the effect of category (stimType) and task relevance (isTaskRelevant)
on a [dependent variable] , in the FIRST time window (0-500ms).

[dependent variable] are 4 different PRE-REGISTERED parameters:
1. Fixation distance from screen center
2. Saccade amplitude
3. Number of blinks
4. Pupil size

For each of these dependent variables, DURING THE FIRST TIME WINDOW, we will
run a linear model with category (faces/objects/letters/false-fonts) and relevance (True/False)
as fixed effects, and subject and stimulus as random intercepts (no random slopes).
The preregistered model assumes interactions.
"

## Prepare data ---------------------

data_firstWindow <- filter(data, timeWindow == "First")


dependent_vars <- c("MedianCenterDistDegs_BLcorrected", "MaxAmp", "NumBlinks", "MedianRealPupilNorm")  # originally: CenterDistDegs (mean, not median) and RealPupilNorm (mean, not median)



## Run the models ---------------------

# hypothesis model
mod_formula <- . ~ stimType * isTaskRelevant + (1 | subCode) + (1 | stimCode)

# nulls
mod_null1 <- . ~ 1 + (1 | subCode) + (1 | stimCode)
mod_null2 <- . ~ stimType + (1 | subCode) + (1 | stimCode)
mod_null3 <- . ~ isTaskRelevant + (1 | subCode) + (1 | stimCode)
nulls <- c(mod_null1, mod_null2, mod_null3)

# loop
for (dep in dependent_vars){
  print("_____________________")
  print(dep)
  print("_____________________")

  # Prepare results saving
  sink(file=paste("model1_", dep, ".txt"), append=TRUE)  # sink will output prints into a txt file instead of a console


  # Hypothesis
  mod_formula_loop <- update(mod_formula, reformulate(".", response = dep))
  model1 <- lmer(mod_formula_loop, data = data_firstWindow)
  model1_summary <- summary(model1)
  print(" _____________________ MODEL 1 SUMMARY  _____________________ ")
  print(model1_summary)  # write to file


  # Effect significance
  print(" _____________________ MODEL 1 EFFECT SIGNIFICANCE  _____________________ ")

  model1_effects <- anova(model1, type = 2)  # this is for testing which fixed effects were significant
  print(model1_effects)


  # Now it's time to perform correction for multiple comparisons. Bonferroni, as per the pre-reg:
  model1_effects[["p_adjust"]] <- p.adjust(model1_effects[["Pr(>F)"]], method = "bonf")
  print(model1_effects)


  # post-hoc for significant effects
  print(" _____________________ MODEL 1 POST HOC  _____________________ ")

  for (i in 1:nrow(model1_effects)) {  # for each effect in the effects table
    print(rownames(model1_effects)[i])
    if (model1_effects[i, "p_adjust"] < 0.05) {  # if it's significant
      if (!grepl(":", rownames(model1_effects)[i])) {
        # if it's a single fixed effect (not an interaction)
        fixed <- rownames(model1_effects)[i]
        em <- emmeans(model1, fixed, lmer.df = "S")
        em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
        print(em_contrast)  # write to file

      } else {  # else, this is a significant interaction effect
        terms <- strsplit(rownames(model1_effects)[i], ":")[[1]]
        fixed1 <- terms[1]
        fixed2 <- terms[2]
        em <- emmeans(model1, c(fixed1,fixed2), lmer.df = "S")
        em_contrast <- contrast(em, method='pairwise', by=fixed2, infer=TRUE, adjust="bonf")
        print(em_contrast)
      }
    }
  }


  # Bayesian
  print(" _____________________ MODEL 1 BAYESIAN  _____________________ ")

  # BIC hypothesis
  bic_model1 <- BIC(model1)
  print(paste("MODEL 1 BIC: ", bic_model1))

  # BIC nulls
  print(paste("------ NULL MODELS: -------"))
  for (null in nulls){
    null_formula_loop <- update(null, reformulate(".", response = dep))
    print("MODEL FORMULA: ")
    print(null_formula_loop)
    model_null <- lmer(null_formula_loop, data = data_firstWindow)
    print("SUMMARY")
    print(summary(model_null))
    print(paste("----- comparison -----"))
    bic_model_null <- BIC(model_null)
    print(paste("BIC: ", bic_model_null))
    # comparison: which model is preferred
    print(bayesfactor_models(model1, denominator = model_null))
    print("MODEL END")
  }


  sink()  # returns output to the console


}






## =========================== ANALYSIS 2 ===========================

"
In this analysis, we take LONG TRIALS ONLY (only trials where duration was 1.5 seconds).

Then, we test the effect of category (stimType), task relevance (isTaskRelevant), AND TIME WINDOW (First, Second, Third)
on the [dependent variable] in each of these conditions.


[dependent variable] are 4 different PRE-REGISTERED parameters:
1. Fixation distance from screen center
2. Saccade amplitude
3. Number of blinks
4. Pupil size

For each of these dependent variables, DURING THE FIRST TIME WINDOW, we will
run a linear model with category (faces/objects/letters/false-fonts) and relevance (True/False)
as fixed effects, and subject and stimulus as random intercepts (no random slopes).
The preregistered model assumes interactions.

"


## Prepare data ---------------------

data_longTrials <- filter(data, plndStimulusDur == 1.5)  # take only long trials



## Run the models ---------------------

mod_formula <- . ~ stimType * isTaskRelevant * timeWindow + (1 | subCode) + (1 | stimCode)

# nulls
mod_null1 <- . ~ 1 + (1 | subCode) + (1 | stimCode)
mod_null2 <- . ~ stimType  + (1 | subCode) + (1 | stimCode)
mod_null3 <- . ~ isTaskRelevant + (1 | subCode) + (1 | stimCode)
mod_null4 <- . ~ timeWindow + (1 | subCode) + (1 | stimCode)
mod_null5 <- . ~ stimType * isTaskRelevant + (1 | subCode) + (1 | stimCode)
mod_null6 <- . ~ isTaskRelevant * timeWindow + (1 | subCode) + (1 | stimCode)
mod_null7 <- . ~ stimType * timeWindow + (1 | subCode) + (1 | stimCode)

nulls <- c(mod_null1, mod_null2, mod_null3, mod_null4, mod_null5, mod_null6, mod_null7)

for (dep in dependent_vars){  # same columns as the first model
  # Prepare results saving
  sink(file=paste("model2_", dep, ".txt"), append=TRUE)

  # Hypothesis
  mod_formula_loop <- update(mod_formula, reformulate(".", response = dep))
  model2 <- lmer(mod_formula_loop, data = data_longTrials)
  model2_summary <- summary(model2)
  print(" _____________________ MODEL 2 SUMMARY  _____________________ ")
  print(model2_summary)  # write to file

  # Effect significance
  print(" _____________________ MODEL 2 EFFECT SIGNIFICANCE  _____________________ ")
  model2_effects <- anova(model2, type = 2)  # this is for testing which fixed effects were significant
  print(model2_effects)

  # Now it's time to perform correction for multiple comparisons. Bonferroni, as per the pre-reg:
  model2_effects[["p_adjust"]] <- p.adjust(model2_effects[["Pr(>F)"]], method = "bonf")
  print(model2_effects)


  # post-hoc for significant effects
  print(" _____________________ MODEL 2 POST HOC  _____________________ ")

  for (i in 1:nrow(model2_effects)) {  # for each effect in the effects table
    if (model2_effects[i, "p_adjust"] < 0.05) {  # if it's significant
      if (!grepl(":", rownames(model2_effects)[i])) {
        # if it's a single fixed effect (not an interaction)
        fixed <- rownames(model2_effects)[i]
        em <- emmeans(model2, fixed, lmer.df = "S")
        em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
        print(em_contrast)

      } else {  # else, this is a significant interaction effect
        terms <- strsplit(rownames(model2_effects)[i], ":")[[1]]
        fixed1 <- terms[1]
        fixed2 <- terms[2]
        em <- emmeans(model2, c(fixed1, fixed2), lmer.df = "S")
        em_contrast <- contrast(em, method='pairwise', by=fixed2, infer=TRUE, adjust="bonf")
        print(em_contrast)

      }
    }
  }

  # Bayesian
  print(" _____________________ MODEL 2 BAYESIAN  _____________________ ")

  # BIC hypothesis
  bic_model2 <- BIC(model2)
  print(paste("MODEL 2 BIC: ", bic_model2))

  # BIC nulls
  print(paste("------ NULL MODELS: -------"))
  for (null in nulls){
    null_formula_loop <- update(null, reformulate(".", response = dep))
    print("MODEL FORMULA: ")
    print(null_formula_loop)
    model_null <- lmer(null_formula_loop, data = data_longTrials)
    print("SUMMARY")
    print(summary(model_null))
    print(paste("----- comparison -----"))
    bic_model_null <- BIC(model_null)
    print(paste("BIC: ", bic_model_null))
    # comparison: which model is preferred
    print(bayesfactor_models(model2, denominator = model_null))
    print("MODEL END")
  }


  sink()  # returns output to the console


}
