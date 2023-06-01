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
The following are the behavioral data analyses for experiment 1 of the Cogitate Consortium.
The models below follow pre-registration 4.0: https://osf.io/gm3vd

The model inputs (data tables) are csv files outputted from the behavior data quality-checks, 
which were done separately in Python; all codes can be found in the consortium's github.

@author: RonyHirsch
"


modalities <- c("fMRI", "MEG", "ECoG")  # all three modalities, this is used for per-modality modelling


## =========================== ANALYSIS 1 =========================== 
"
In this analysis, we test the effect of category (stimType) and duration group (plndStimulusDur) on sensitivity (dPrime).
We run a linear model with category (faces/objects/letters/false-fonts), duration (0.5/1.0/1.5 s) and modality (M-EEG, fMRI, iEEG) as fixed effects 
and subject as a random intercept. The preregistered model assumes interactions. 
"

## Prepare results saving ---------------------

sink(file="model1.txt", append=TRUE)  # sink will output prints into a txt file instead of a console


## Data ---------------------

data <- read.csv("lmm_dprime_cat_dur_mod_phase3.csv")   
# dPrime is our dependent variable (number), but all the rest are categorical
data$subCode <- factor(data$subCode)  # subject code
data$modality <- factor(data$modality)  # fMRI, MEG, ECoG
data$stimType <- factor(data$stimType)  # face, object, letter, falseFont
data$plndStimulusDur <- factor(data$plndStimulusDur)  # 0.5, 1, 1.5 (seconds)



## Model ------------------------

# Hypothesis model
"
We use afex lmer_alt instead of lmerTest's lmer to introduce random slopes without the covariances among them (expand_re=TRUE).  
Otherwise, we would recieve an error due to the number of parameters // would have to resort to a linear model w/o random slopes i.e.
lmer(dPrime ~ modality * stimType * plndStimulusDur + (1|subCode), data=data)
"
model1_h1 <- afex::lmer_alt(dPrime ~ modality * stimType * plndStimulusDur + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
model1_h1_summary <- summary(model1_h1)  # see how much the dPrime changes with per fixed effect units, summarize the model
# write to file
print(" _____________________ MODEL 1 SUMMARY  _____________________ ")
print(model1_h1_summary)  


## Effect Significance ------------------------

print(" _____________________ MODEL 1 EFFECT SIGNIFICANCE  _____________________ ")

model1_h1_effects <- anova(model1_h1, type = 2)  # this is for testing which fixed effects were significant
print(model1_h1_effects)  # write to file

# Now it's time to perform correction for multiple comparisons. Bonferroni, as per the pre-reg:
model1_h1_effects[["p_adjust"]] <- p.adjust(model1_h1_effects[["Pr(>F)"]], method = "bonf")
print(model1_h1_effects)  # write to file


## Post-Hoc ------------------------

"
Pre-reg 4.0:
'In all analyses, post-hoc comparisons will follow significant interactions using pairwise t-tests, with Bonferroni correction'
For each significant effect we found above, we will compute the estimated marginal means (least-square means).
Then, we will use the contrast method to obtain a pairwise contrasts among estimators.
"

print(" _____________________ MODEL 1 POST-HOC  _____________________ ")

em_mod <- emmeans(model1_h1, ~modality, lmer.df = "S") # without 'lmer.df = "S"' it crashes
em_mod_contrast <- contrast(em_mod, method='pairwise',infer=TRUE, adjust="bonf")  # the default correction for multiple comparisons is Tukey, we use Bonferroni to follow the pre-reg
print(em_mod_contrast)  # write to file

em_cat <- emmeans(model1_h1, ~stimType, lmer.df = "S")
em_cat_contrast <- contrast(em_cat, method='pairwise',infer=TRUE, adjust="bonf")
print(em_cat_contrast)  # write to file

em_dur <- emmeans(model1_h1, ~plndStimulusDur, lmer.df = "S")
em_dur_contrast <- contrast(em_dur, method='pairwise',infer=TRUE, adjust="bonf")
print(em_dur_contrast)  # write to file

em_modxcat <- emmeans(model1_h1, ~modality+stimType, lmer.df = "S")  
em_modxcat_contrast <- contrast(em_modxcat, method='pairwise', by='modality', infer=TRUE, adjust="bonf")
print(em_modxcat_contrast)  # write to file


## Bayesian ------------------------
"
Pre-reg 4.0:
'all..analysis will be complemented by a Bayesian LMM withthe same parameters. 
We will use JZS Bayes Factor with a Cauchy prior andone scaling parameter ‘r’ of sqrt(2)/2 = 0.7071. 
For each effect of interest, we will compare the tested model with a matching null-hypothesis model that excludes the effect of interest.'

Therefore: we follow the pre-reg, using BayesFactor package, which uses the JZS prior:
Rouder, J.N., Speckman, P.L., Sun, D. et al. Bayesian t tests for accepting and rejecting the null hypothesis. 
Psychonomic Bulletin & Review 16, 225–237 (2009). https://doi.org/10.3758/PBR.16.2.225

This returns a table where we have BF that compares between the full model and each model sans one of the effects. 
"

# nulls

mod_null1 <- afex::lmer_alt(dPrime ~ 1 + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
mod_null2 <- afex::lmer_alt(dPrime ~ modality + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
mod_null3 <- afex::lmer_alt(dPrime ~ stimType + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
mod_null4 <- afex::lmer_alt(dPrime ~ plndStimulusDur + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
mod_null5 <- afex::lmer_alt(dPrime ~ modality * stimType + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
mod_null6 <- afex::lmer_alt(dPrime ~ stimType * plndStimulusDur + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
mod_null7 <- afex::lmer_alt(dPrime ~ modality * plndStimulusDur + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
nulls <- c(mod_null1, mod_null2, mod_null3, mod_null4, mod_null5, mod_null6, mod_null7)


# BIC hypothesis
bic_model1_h1 <- BIC(model1_h1)
print(paste("MODEL 1H1 BIC: ", bic_model1_h1))

# BIC nulls
print(paste("------ NULL MODELS: -------"))
for (null in nulls){
  print("----- model -----")
  model_null <- null
  print(null)
  print(paste("----- comparison -----"))
  bic_model_null <- BIC(model_null)
  print(paste("BIC: ", bic_model_null))
  # comparison: which model is preferred
  print(bayesfactor_models(model1_h1, denominator = model_null))
  print("MODEL END")
}





## ============= ANALYSIS 1: PER LAB ============= 


for (modal in modalities){
  mod_data <- data[data$modality == modal, ]
  print(paste("------------- PER MODALITY: ", modal, " -------------"))
  # this time the model is PER LAB, not PER SITE
  
  model <- afex::lmer_alt(dPrime ~ lab * stimType * plndStimulusDur + ((lab + stimType + plndStimulusDur)^2 || subCode), 
                          data = mod_data, expand_re = TRUE)
  model_summary <- summary(model)
  print(paste(" ____________ MODEL 1 SUMMARY: ",   modal,"____________ "))
  print(model_summary) 
  
  print(paste(" ____________ MODEL 1 EFFECT SIGNIFICANCE: ",   modal,"____________ "))
  model_effects <- anova(model, type = 2)  
  print(model_effects) 
  model_effects[["p_adjust"]] <- p.adjust(model_effects[["Pr(>F)"]], method = "bonf")
  print(model_effects)  
  
  print(paste(" ____________ MODEL 1 POST HOC: ",   modal,"____________ "))
  em <- emmeans(model, ~lab, lmer.df = "S") 
  em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf") 
  print(em_contrast) 
  
  em <- emmeans(model, ~stimType, lmer.df = "S")
  em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
  print(em_contrast) 
  
  em <- emmeans(model, ~plndStimulusDur, lmer.df = "S")
  em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
  print(em_contrast)  
  
  em <- emmeans(model, ~lab+stimType, lmer.df = "S")  
  em_contrast <- contrast(em, method='pairwise', by='lab', infer=TRUE, adjust="bonf")
  print(em_contrast) 
 
  
}



sink()  # returns output to the console





## =========================== ANALYSIS 1 supplement =========================== 
"
In this analysis, we test the effect of category (stimType) and duration group (plndStimulusDur) on HIT RATE!!.
This is the same model as 1, but with hit rates instead of the corrected d'. 
This is because the d' has artificial differences, due to the different number of trials across modalities. 
"

## Prepare results saving ---------------------

sink(file="model1_supp.txt", append=TRUE)  # sink will output prints into a txt file instead of a console


## Data ---------------------

data <- read.csv("lmm_dprime_cat_dur_mod_phase3.csv")   
# dPrime is our dependent variable (number), but all the rest are categorical
data$subCode <- factor(data$subCode)  # subject code
data$modality <- factor(data$modality)  # fMRI, MEG, ECoG
data$stimType <- factor(data$stimType)  # face, object, letter, falseFont
data$plndStimulusDur <- factor(data$plndStimulusDur)  # 0.5, 1, 1.5 (seconds)


model1s_h1 <- afex::lmer_alt(hit_rate ~ modality * stimType * plndStimulusDur + ((modality + stimType + plndStimulusDur)^2 || subCode), data = data, expand_re = TRUE, control = lmerControl("bobyqa"))
model1s_h1_summary <- summary(model1s_h1)  
print("")
print(" _____________________ MODEL 1 SUMMARY  _____________________ ")
print(model1s_h1_summary)  

print("")
print(" _____________________ MODEL 1 EFFECT SIGNIFICANCE  _____________________ ")
model1s_h1_effects <- anova(model1s_h1, type = 2)  
print(model1s_h1_effects) 

model1s_h1_effects[["p_adjust"]] <- p.adjust(model1s_h1_effects[["Pr(>F)"]], method = "bonf")
print(model1s_h1_effects) 

print("")
print(" _____________________ MODEL 1 POST-HOC  _____________________ ")
em_mod <- emmeans(model1s_h1, ~modality, lmer.df = "S") # without 'lmer.df = "S"' it crashes
em_mod_contrast <- contrast(em_mod, method='pairwise',infer=TRUE, adjust="bonf")  # the default correction for multiple comparisons is Tukey, we use Bonferroni to follow the pre-reg
print(em_mod_contrast)  # write to file

em_cat <- emmeans(model1s_h1, ~stimType, lmer.df = "S")
em_cat_contrast <- contrast(em_cat, method='pairwise',infer=TRUE, adjust="bonf")
print(em_cat_contrast)  # write to file

em_dur <- emmeans(model1s_h1, ~plndStimulusDur, lmer.df = "S")
em_dur_contrast <- contrast(em_dur, method='pairwise',infer=TRUE, adjust="bonf")
print(em_dur_contrast)  # write to file

em_modxcat <- emmeans(model1s_h1, ~modality+stimType, lmer.df = "S")  
em_modxcat_contrast <- contrast(em_modxcat, method='pairwise', by='modality', infer=TRUE, adjust="bonf")
print(em_modxcat_contrast)  # write to file

sink()  # returns output to the console








## =========================== ANALYSIS 2 =========================== 
"
In this analysis, we test the effect of category (stimType) and task relevance (isTaskRelevant) on false alarm rates.
Pre-reg 4.0:
'Modality will be included as a factor to detect potential effects that are uniquely present in one of the modalities.
We expect to find more FAs for task-relevant than for task-irrelevant stimuli, 
indicating that subjects kept a category template in mind and did not apply it to irrelevant categories.'

Notably, as there are very few false-alarm trials and multiple effects, 
the original model (including random intercept per stimulus) failed to converge. 
Therefore, we model as per the pre-registration's fixed effects, with subject as
the only random intercept. 
"

## Prepare results saving ---------------------

sink(file="model2.txt", append=TRUE)  # sink will output prints into a txt file instead of a console



## Data ---------------------

data <- read.csv("lmm_fa_cat_rt_phase3.csv")
# categorical fixed effects
data$subCode <- factor(data$subCode)  # subject code
data$modality <- factor(data$modality)  # fMRI, MEG, ECoG
data$stimType <- factor(data$stimType)  # face, object, letter, falseFont
data$isTaskRelevant <- factor(data$isTaskRelevant)  # True, False

# dependent variable: false alarm (is it a false alarm, yes or no)
data$isFalsePositive <- factor(data$isFalsePositive)  # True, False



## Model ------------------------

# prepare dataframe for modelling
data_summ <- data |> 
  group_by(modality, stimType, isTaskRelevant, subCode) |> 
  summarise(N = n(),
            k = sum(isFalsePositive == "True")) |> 
  ungroup()
write.csv(data_summ, "model2_data.csv", row.names=FALSE)  # save modeled data to csv


# Hypothesis
model2_h1 <- glmer(cbind(k, N-k) ~ modality * stimType * isTaskRelevant + (1 | subCode), data = data_summ, family = binomial("logit"), control = glmerControl("bobyqa"))
model2_h1_summary <- summary(model2_h1)
print("")
print(" _____________________ MODEL 2 SUMMARY  _____________________ ")
print(model2_h1_summary)  # write to file
saveRDS(model2_h1, file = "model2_h1.rds")



## Effect Significance ------------------------
print("")
print(" _____________________ MODEL 2 EFFECT SIGNIFICANCE  _____________________ ")

model2_h1_effects <- car::Anova(model2_h1, type = 2)  # this is for testing which fixed effects were significant; "car" is so we will have the p-values
print(model2_h1_effects)  # write to file

# Now it's time to perform correction for multiple comparisons. Bonferroni, as per the pre-reg:
model2_h1_effects[["p_adjust"]] <- p.adjust(model2_h1_effects[["Pr(>Chisq)"]], method = "bonf")
print(model2_h1_effects)  # write to file


# Post-Hoc ------------------------

"
Pre-reg 4.0:
'In all analyses, post-hoc comparisons will follow significant interactions using pairwise t-tests, with Bonferroni correction'
For each significant effect we found above, we will compute the estimated marginal means (least-square means).
Then, we will use the contrast method to obrain a pairwise contrasts among estimators.
"
print("")
print(" _____________________ MODEL 2 POST HOC  _____________________ ")

em_mod2 <- emmeans(model2_h1, ~modality, lmer.df = "S", type = "response")
print(em_mod2)

em_mod2_contrast <- contrast(em_mod2, method='pairwise',infer=TRUE, adjust="bonf")
print(em_mod2_contrast)  # write to file

em_cat2 <- emmeans(model2_h1, ~stimType, lmer.df = "S", type = "response")
print(em_cat2)
em_cat2_contrast <- contrast(em_cat2, method='pairwise',infer=TRUE, adjust="bonf")
print(em_cat2_contrast)  # write to file

em_tr2 <- emmeans(model2_h1, ~isTaskRelevant, lmer.df = "S", type = "response")
print(em_tr2)
em_tr2_contrast <- contrast(em_tr2, method='pairwise',infer=TRUE, adjust="bonf")
print(em_tr2_contrast)  # write to file

em_modxcat2 <- emmeans(model2_h1, ~modality+stimType, lmer.df = "S", type = "response")  
print(em_modxcat2)
em_modxcat2_contrast <- contrast(em_modxcat2, method='pairwise', by='modality', infer=TRUE, adjust="bonf")
print(em_modxcat2_contrast)  # write to file

em_modxtr2 <- emmeans(model2_h1, ~modality+isTaskRelevant, lmer.df = "S", type = "response")  
print(em_modxtr2)
em_modxtr2_contrast <- contrast(em_modxtr2, method='pairwise', by='modality', infer=TRUE, adjust="bonf")
print(em_modxtr2_contrast)  # write to file

em_catxtr2 <- emmeans(model2_h1, ~stimType+isTaskRelevant, lmer.df = "S", type = "response")  
print(em_catxtr2)
em_catxtr2_contrast <- contrast(em_catxtr2, method='pairwise', by='isTaskRelevant', infer=TRUE, adjust="bonf")
print(em_catxtr2_contrast)  # write to file

em_all2 <- emmeans(model2_h1, ~modality+stimType+isTaskRelevant, lmer.df = "S", type = "response") 
print(em_all2)
em_all2_contrast <- contrast(em_all2, method='pairwise', by='modality', infer=TRUE, adjust="bonf")
print(em_all2_contrast)  # write to file



# Bayesian ------------------------
"
Pre-reg 4.0:
'all..analysis will be complemented by a Bayesian LMM withthe same parameters. 
We will use JZS Bayes Factor with a Cauchy prior andone scaling parameter ‘r’ of sqrt(2)/2 = 0.7071. 
For each effect of interest, we will compare the tested model with a matching null-hypothesis model that excludes the effect of interest.'

HOWEVER, model 2 is not a linear model, and so using Bayes Factor is not right here. 
Therefore, we will use p_to_bf. For each p-value, we will recieve an approximation of the BF. 
https://easystats.github.io/bayestestR/reference/p_to_bf.html

We will provide this function tieh the ORIGINAL P-VALUES, BEFORE BONFERRONI CORRECTION
"
print("")
print(" _____________________ MODEL 2 BAYESIAN  _____________________ ")

print("Num of obs")
print(nrow(data_summ))
print("")

BFModles2 <- p_to_bf(model2_h1_effects[["Pr(>Chisq)"]] , n_obs=nrow(data_summ)) 
print(BFModles2)



## ============= ANALYSIS 2: PER LAB ============= 


for (modal in modalities){
  print(paste("------------- PER MODALITY: ", modal, " -------------"))
  mod_data <- data[data$modality == modal, ]
  mod_data_summ <- mod_data |> 
    group_by(lab, stimType, isTaskRelevant, subCode) |> 
    summarise(N = n(), k = sum(isFalsePositive == "True")) |> 
    ungroup()
  # save to file
  write.csv(mod_data_summ, paste(modal, "_model2_per_lab_data.csv"), row.names=FALSE)  # save data to csv
  
  # this time the model is PER LAB, not PER SITE
  
  model <- glmer(cbind(k, N-k) ~ lab * stimType * isTaskRelevant + (1 | subCode), data = mod_data_summ, family = binomial("logit"), 
                 contrasts = list(isTaskRelevant = contr.sum, lab = contr.sum, stimType = contr.sum))  # this helps estimating the standard errors, as FA rates are actually too low to correctly run the model
  model_summary <- summary(model)
  print(paste(" ____________ MODEL 2 SUMMARY: ",   modal,"____________ "))
  print(model_summary) 
  
  
  print(paste(" ____________ MODEL 2 EFFECT SIGNIFICANCE: ",   modal,"____________ "))
  model_effects <- car::Anova(model, type = 2)  
  print(model_effects)  
  
  # Now it's time to perform correction for multiple comparisons. Bonferroni, as per the pre-reg:
  model_effects[["p_adjust"]] <- p.adjust(model_effects[["Pr(>Chisq)"]], method = "bonf")
  print(model_effects)  
  
  
}




sink()  # returns output to the console






## =========================== ANALYSIS 3 =========================== 
"
Pre-reg 4.0:
'Subjects reaction times (RT) will also be examined, using  the first LMM  described above.'
I interpreted this as testing the effect of category and duration on participants' reaction times for hit responses.
We run a linear model with category (faces/objects/letters/false-fonts), duration (0.5/1.0/1.5 s) 
and modality (M-EEG, fMRI, iEEG) as fixed effects and subject and item as random intercepts.
"

## Prepare results saving ---------------------

sink(file="model3.txt", append=TRUE)  # sink will output prints into a txt file instead of a console


## Data ---------------------

data <- read.csv("lmm_hitRTs_cat_dur_mod_trials_phase3.csv")
# reactionTime is our dependent variable (number), but all the rest are categorical
data$subCode <- factor(data$subCode)  # subject code
data$modality <- factor(data$modality)  # fMRI, MEG, ECoG
data$stimType <- factor(data$stimType)  # face, object, letter, falseFont
data$plndStimulusDur <- factor(data$plndStimulusDur)  # 0.5, 1, 1.5 (seconds)
data$stimCode <- factor(data$stimCode)  # stimulus code (unique per item)



## Model ------------------------

# Hypothesis
model3_h1 <- lmer(reactionTime ~ modality * stimType * plndStimulusDur + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
model3_h1_summary <- summary(model3_h1)
print(" _____________________ MODEL 3 SUMMARY  _____________________ ")
print(model3_h1_summary)  # write to file



## Effect Significance ------------------------

print(" _____________________ MODEL 3 EFFECT SIGNIFICANCE  _____________________ ")

model3_h1_effects <- anova(model3_h1, type = 2)  # this is for testing which fixed effects were significant
print(model3_h1_effects)  # write to file

# Now it's time to perform correction for multiple comparisons. Bonferroni, as per the pre-reg:
model3_h1_effects[["p_adjust"]] <- p.adjust(model3_h1_effects[["Pr(>F)"]], method = "bonf")
print(model3_h1_effects)  # write to file


# Post-Hoc ------------------------

"
Pre-reg 4.0:
'In all analyses, post-hoc comparisons will follow significant interactions using pairwise t-tests, with Bonferroni correction'
For each significant effect we found above, we will compute the estimated marginal means (least-square means).
Then, we will use the contrast method to obrain a pairwise contrasts among estimators.
"

print(" _____________________ MODEL 3 POST HOC  _____________________ ")

em_mod3 <- emmeans(model3_h1, ~modality, lmer.df = "S") 
em_mod3_contrast <- contrast(em_mod3, method='pairwise',infer=TRUE, adjust="bonf")
print(em_mod3_contrast)  # write to file

em_cat3 <- emmeans(model3_h1, ~stimType, lmer.df = "S")
em_cat3_contrast <- contrast(em_cat3, method='pairwise',infer=TRUE, adjust="bonf")
print(em_cat3_contrast)  # write to file

em_modxcat3 <- emmeans(model3_h1, ~modality+stimType, lmer.df = "S")  
em_modxcat3_contrast <- contrast(em_modxcat3, method='pairwise', by='modality', infer=TRUE, adjust="bonf")
print(em_modxcat3_contrast)  # write to file



# Bayesian ------------------------
"
Pre-reg 4.0:
'all..analysis will be complemented by a Bayesian LMM withthe same parameters. 
We will use JZS Bayes Factor with a Cauchy prior andone scaling parameter ‘r’ of sqrt(2)/2 = 0.7071. 
For each effect of interest, we will compare the tested model with a matching null-hypothesis model that excludes the effect of interest.'

Therefore: we follow the pre-reg, using BayesFactor package, which uses the JZS prior:
Rouder, J.N., Speckman, P.L., Sun, D. et al. Bayesian t tests for accepting and rejecting the null hypothesis. 
Psychonomic Bulletin & Review 16, 225–237 (2009). https://doi.org/10.3758/PBR.16.2.225

This returns a table where we have BF that compares between the full model and each model sans one of the effects. 
"

# nulls

mod_null1 <- lmer(reactionTime ~ 1 + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
mod_null2 <- lmer(reactionTime ~ modality  + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
mod_null3 <- lmer(reactionTime ~ stimType + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
mod_null4 <- lmer(reactionTime ~ plndStimulusDur + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
mod_null5 <- lmer(reactionTime ~ modality * stimType + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
mod_null6 <- lmer(reactionTime ~ stimType * plndStimulusDur + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
mod_null7 <- lmer(reactionTime ~ modality * plndStimulusDur + (1 | subCode) + (1 | stimCode), data = data, control = lmerControl("bobyqa"))
nulls <- c(mod_null1, mod_null2, mod_null3, mod_null4, mod_null5, mod_null6, mod_null7)


# BIC hypothesis
bic_model3_h1 <- BIC(model3_h1)
print(paste("MODEL 3H1 BIC: ", bic_model3_h1))

# BIC nulls
print(paste("------ NULL MODELS: -------"))
for (null in nulls){
  model_null <- null
  print(null)
  print(paste("----- comparison -----"))
  bic_model_null <- BIC(model_null)
  print(paste("BIC: ", bic_model_null))
  # comparison: which model is preferred
  print(bayesfactor_models(model3_h1, denominator = model_null))
  print("MODEL END")
}





## ============= ANALYSIS 3: PER LAB ============= 


for (modal in modalities){
  mod_data <- data[data$modality == modal, ]
  print(paste("------------- PER MODALITY: ", modal, " -------------"))
  # this time the model is PER LAB, not PER SITE
  
  model <- lmer(reactionTime ~ lab * stimType * plndStimulusDur + (1 | subCode) + (1 | stimCode), data = mod_data)
  model_summary <- summary(model)
  print(paste(" ____________ MODEL 3 SUMMARY: ",   modal,"____________ "))
  print(model_summary) 
  
  print(paste(" ____________ MODEL 3 EFFECT SIGNIFICANCE: ",   modal,"____________ "))
  model_effects <- anova(model, type = 2)  
  print(model_effects) 
  model_effects[["p_adjust"]] <- p.adjust(model_effects[["Pr(>F)"]], method = "bonf")
  print(model_effects)  
  
  
}



sink()  # returns output to the console
