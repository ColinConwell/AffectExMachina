if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('this.path', 'mgsub', 'arrow', 'scales', 'weights', 'psych', 'mgcv', 'skimr',
               'ggeasy', 'ggpubr', 'ggfortify', 'ggstatsplot', 'ggExtra','ggcorrplot',
               'cowplot', 'magrittr', 'stargazer', 'rstatix', 'tidymodels', 'tidytext', 'tidyverse')

setwd(dirname(this.path()))

#https://rpubs.com/dgolicher/median_boot
ci_norm <- function(x) {return(qnorm(0.975) * (sd(x) / sqrt(n())))}
ci_boot <- function(x) {return(mean_cl_boot(x, conf.int = 0.95) %>% 
                                 mutate(ci=ymax-y) %>% pull(ci))}

custom_themes <- list()
theme_set(theme_bw())

median_cl_boot <- function(x, conf = 0.95) {
  lconf <- (1 - conf)/2
  uconf <- 1 - lconf
  require(boot)
  bmedian <- function(x, ind) median(x[ind])
  bt <- boot(x, bmedian, 1000)
  bb <- boot.ci(bt, type = "perc")
  data.frame(y = median(x), ymin = quantile(bt$t, lconf), 
             ymax = quantile(bt$t, uconf))
}

### Specifications ------------------------------------------------------

custom_themes[['bottom_legend']] <- theme(legend.position="bottom", legend.justification="center", 
                                          legend.box.margin=margin(-12,0,0,0))

custom_themes[['top_legend']] <- theme(legend.position="top", legend.justification="center", 
                                          legend.box.margin=margin(-12,0,0,0))

reverse_labels <- function(df) {enframe(df) %>% select(2:1) %>% deframe}

measurement_levels <- c('Arousal','Valence','Beauty')

oasis_labels <- c(Animal = 'Animal', Person = 'Person', 
                  Object = 'Object', Scene = 'Scene', Combo = 'Combo')
vessel_labels <- c(Art = 'art', Buildings = 'are', Interiors = 'ari', 
                   'Faces' = 'fac', Landscapes = 'lsc', Combo = 'Combo')
image_type_labels <- c(oasis_labels[-5], vessel_labels)

oasis_levels <- names(oasis_labels)
vessel_levels <- unname(reverse_labels(vessel_labels))
image_type_levels <- c(oasis_levels[-5], vessel_levels)

fix_factor_levels <- function(x) {
  x %<>% mutate(dataset = factor(str_to_title(dataset)),
                image_type = factor(image_type),
                measurement = factor(measurement))
  
  dataset_levels <- c(sort(levels(x$dataset)))
  current_levels <- levels(x$image_type)
  measurement_levels_ <- measurement_levels
  if (identical(c('Oasis','Vessel'), dataset_levels)) {
    new_labels <- image_type_labels
    new_levels <- image_type_levels
  } else if (identical(c('Oasis'), dataset_levels)) {
    new_labels <- oasis_labels
    new_levels <- oasis_levels
  } else if (identical(c('Vessel'), dataset_levels)) {
    new_labels <- vessel_labels
    new_levels <- vessel_levels
    measurement_levels_ <- 'Beauty'
  } else { # catch all
    print('Dataset match not found!')
  } # conditionals by dataset
  
  if (!has_element(current_levels, 'lsc')) {new_labels <- new_levels; names(new_labels) <- new_levels}
  
  x %>% mutate(measurement = factor(str_to_title(measurement), measurement_levels_)) %>%
    mutate(image_type = fct_recode(image_type, !!!new_labels),
           image_type = factor(image_type, levels = new_levels))
}

self_supervised_regex <- 'selfsupervised|seer'
detectron_regex <- 'segmentation|detection|panoptics'

model_typology <- read_csv('model_opts/model_typology.csv') %>%
  mutate(task_cluster = ifelse(model == 'denoising', '2D', task_cluster))
model_display_names <- model_typology %>% distinct(model, model_display_name)

### Response Data --------------------------------------------------

oasis_responses <- read_csv('response/oasis_means_per_image.csv')


### Reliability ----------------------------------------------------

mmo_corrs <- read_csv('response/vessel_oracle_data.csv') %>%
  mutate(measurement = 'beauty', dataset = 'vessel') %>%
  bind_rows(read_csv('response/oasis_oracle_data.csv') %>%
              rename(image_type = category) %>%
              mutate(dataset = 'oasis')) %>%
  mutate(corr = oracle_corr) %>% select(-oracle_corr) %>%
  rename(image_count = item_count) %>% fix_factor_levels()

mmo_summary <- mmo_corrs %>%
  group_by(dataset, image_type, measurement) %>% 
  summarise(ci = ci_boot(corr), r = mean(corr, na.rm = TRUE),
            min = min(corr, na.rm = TRUE), 
            max = max(corr, na.rm = TRUE),
            se = sd(corr, na.rm = TRUE) / sqrt(n()))

splithalf_summary <- read_csv('response/splithalf_data.csv') %>% fix_factor_levels() %>%
  rename(r = splithalf_r, lower = splithalf_lower, upper = splithalf_upper)

ggplot(mmo_summary, aes(image_type, r, fill = measurement)) + 
  facet_wrap(~dataset, scales = 'free_x') +
  geom_bar(stat = 'identity', position = "dodge2") +
  geom_errorbar(aes(ymin = r - ci, ymax = r + ci), 
                position = position_dodge(width = 0.9), width = 0.1)

ggplot(splithalf_summary, aes(image_type, r, fill = measurement)) + 
  facet_wrap(~dataset, scales = 'free_x') +
  geom_bar(stat = 'identity', position = "dodge2") +
  geom_errorbar(aes(ymin = lower, ymax = upper), 
                position = position_dodge(width = 0.9), width = 0.1)

reliability_combo <- mmo_summary %>%
  select(measurement, image_type, dataset, ci, r) %>%
  group_by(measurement, image_type, dataset) %>%
  summarise(r = r, lower = r - ci, upper = r + ci) %>%
  ungroup %>% mutate(reliability = 'Mean-Minus-One') %>%
  bind_rows(splithalf_summary %>% mutate(reliability = 'Split-Half') %>%
              select(dataset, measurement, image_type, r, upper, lower, reliability)) %>%
  mutate(measurement = factor(measurement, measurement_levels))

levels(reliability_combo$measurement)
levels(reliability_combo$dataset)
levels(reliability_combo$image_type)

### Regression Data --------------------------------------------------

read_csv_add_dataset <- function(x) {read_csv(x, col_types = cols()) %>% 
    mutate(dataset = str_split(x, '/', simplify = TRUE) %>% nth(-2))}

reg_results <- dir('incoming/reg_redux', pattern = '.csv', full.names = TRUE, recursive = TRUE) %>%
  map(read_csv_add_dataset) %>% bind_rows() %>% fix_factor_levels() %>%
  filter(model != 'alexnet', score_type == 'pearson_r') %>%
  mutate(train_type = str_replace(train_type, detectron_regex, 'detectron')) %>%
  group_by(model, train_type) %>%
  mutate(model_depth = n_distinct(model_layer_index),
         model_layer_depth = model_layer_index / model_depth) %>% ungroup()

reg_results_max <- reg_results %>%
  group_by(model, train_type, dataset, 
           measurement, image_type) %>% 
  filter(score == max(score)) %>% ungroup()

levels(reg_results_max$measurement)
levels(reg_results_max$dataset)
levels(reg_results_max$image_type)

read_parquet_add_dataset <- function(x) {read_parquet(x) %>% 
    mutate(dataset = str_split(x, '/', simplify = TRUE) %>% nth(-2))}

boot_results <- dir('incoming/bootstrapping', pattern = '.parquet', full.names = TRUE, recursive = TRUE) %>%
  map(read_parquet_add_dataset) %>% bind_rows() %>% rename(bootstrap_id = bootstrap_ids) %>%
  select(-`__index_level_0__`, -model_layer_index) %>%
  filter(model != 'alexnet', score_type == 'pearson_r', bootstrap_id < 1000) %>% fix_factor_levels() %>%
  mutate(train_type = str_replace(train_type, detectron_regex, 'detectron'))

### General Rankings -----------------------------------------

# ranking models by their average score across measurements
reg_results_max %>% group_by(model, train_type) %>%
  summarise(n = n(), score_ci = mean_cl_boot(score)) %>% 
  ungroup() %>% unnest(score_ci) %>% filter(n == 21) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  mutate(rank = dense_rank(-y)) %>% arrange(rank) %>%
  filter(row_number() %% 1 == 0) %>% print(n = 250)

# averaging scores across layers, filtering the highest scoring layer ...
reg_results %>% #filter(!train_type %in% c('clip')) %>%
  #filter(measurement == 'Beauty') %>%
  group_by(model, train_type, model_layer, model_layer_depth) %>%
  summarise(score = mean(score)) %>%
  group_by(model, train_type) %>%
  filter(score == max(score)) %>% 
  arrange(desc(score)) %T>% print() %>%
  mutate(model_string = paste(model, train_type, sep = '_')) %>%
  write.csv('superlative_layers.csv', row.names = FALSE)
# for use in later experiments (e.g. cross-decoding)

# filtering the best model by train_type
reg_results_max %>% group_by(model, train_type) %>%
  summarise(n = n(), score_ci = mean_cl_boot(score)) %>% 
  ungroup() %>% unnest(score_ci) %>% 
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  group_by(train_type) %>% mutate(rank = dense_rank(-y)) %>% 
  arrange(rank) %>% filter(rank == 1) %>% arrange(desc(y))

# bootstrapped rankings of models, averaged
boot_results %>% filter(image_type == 'Combo') %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = dense_rank(-score)) %>% arrange(rank) %>%
  select(model, train_type, score, lower_ci, upper_ci, rank) %>% print(n = 200)

# bootstrapped rankings of models in explained_variance, averaged
boot_results %>% filter(image_type == 'Combo') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = dense_rank(-score)) %>% arrange(rank) %>%
  mutate_at(vars(score, lower_ci, upper_ci), round, 3) %>%
  mutate(report = paste0(score,' [',lower_ci,', ',upper_ci,']')) %>%
  select(model, train_type, report) %>% print(n = 200)

# best layers by measurement / image_type for example model
reg_results %>% filter(!train_type %in% c('clip')) %>%
  group_by(model, train_type, dataset, measurement, image_type) %>%
  filter(score == max(score)) %>% ungroup() %>%
  filter(image_type %in% c('Landscapes','Scene','Combo'), measurement == 'Beauty',
         model == 'vit_base_patch16_224', train_type == 'imagenet') %>% select(-alpha, -score_type)

### Section: Methods -------------------------------------------------

# How many models are in our survey?
reg_results %>% select(model, train_type) %>% n_distinct()
reg_results %>% select(model) %>% n_distinct()

### Results 3.1 Overall Accuracy -------------------------------------------------

# Overall scores for ImageNet-trained models, in Pearson R
reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  summarise(ci = list(mean_cl_normal(score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  filter(train_type == 'imagenet', image_type == 'Combo')

# Reliability estimates: mean, ci
reliability_combo %>% filter(image_type == 'Combo') %>%
  mutate_at(vars(r, lower, upper), round, 3) %>%
  mutate(report = paste0(r,' [',lower,', ',upper,']'))

# percentage of Imagenet models more predictive than taste-typical subjects
mmo_corrs %>% group_by(dataset, image_type, measurement) %>%
  summarise(quantile = scales::percent(seq(0.0,1,0.01)),
            corr = quantile(corr, seq(0.0,1,0.01), na.rm = TRUE)) %>%
  left_join(reg_results_max %>% filter(train_type == 'imagenet') %>%
              group_by(train_type, measurement, image_type, dataset) %>%
              summarise(score = mean(score))) %>% mutate(diff = abs(score - corr)) %>%
  filter(diff == min(diff)) %>% filter(image_type == 'Combo')
  
# Overall explainable variance explained for Imagenet models
reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  summarise(ci = list(mean_cl_normal(explained_variance))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  filter(train_type %in% c('imagenet'), image_type == 'Combo') %T>%
  print() %>% ungroup() %>% summarise(var_explained = mean(y))
    
### Results 3.2 Superlative Models -------------------------------------------------
    
# bootstrapped rankings of models, averaged
boot_results %>% filter(image_type == 'Combo') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance, na.rm = TRUE)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = dense_rank(explained_variance)) %>% arrange(rank) %>%
  mutate_at(vars(explained_variance, lower_ci, upper_ci), round, 3) %>%
  mutate(report = paste0(explained_variance,' [',lower_ci,', ',upper_ci,']')) %>%
  select(model, train_type, report, rank) %>% print(n = 200)    
    
### Results 3.3 Random Networks -------------------------------------------------

# The difference between trained and random models
reg_results_max %>% filter(train_type %in% c('imagenet','random')) %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  grouped_ggwithinstats(x = train_type, y = score, 
                        grouping.var = measurement, type = 'parametric')

reg_results_max %>% filter(train_type %in% c('imagenet','random')) %>%
  group_by(measurement, image_type, dataset) %>%
  {left_join(rstatix::t_test(., score ~ train_type) %>% adjust_pvalue(method = 'holm'),
             rstatix::cohens_d(., score ~ train_type, hedges.correction = TRUE))} %>% 
  filter(p.adj < 0.001) %>% summarise(mean_hedges_g = mean(abs(effsize)))

### Results 3.4 Taskonomy Results -------------------------------------------------

# taskonomy scores for the 'Scene' category
reg_results_max %>% filter(train_type == 'taskonomy') %>%
  group_by(model, train_type, image_type) %>%
  summarise(score = mean(score)) %>%
  filter(image_type == 'Scene') %>%
  arrange(desc(score)) %>% print(n=25)

# bootstrapped taskonomy rankings
boot_results %>% filter(train_type == 'taskonomy') %>%
  filter(image_type %in% c('Combo')) %>%
  group_by(dataset, measurement, image_type, bootstrap_id) %>%
  mutate(rank = dense_rank(-score)) %>%
  group_by(model, train_type, dataset, measurement, image_type, rank) %>%
  summarise(rank_count = n()) %>%
  filter(rank <= 2) %>% arrange(measurement)

boot_results %>% filter(train_type == 'taskonomy') %>%
  filter(image_type %in% c('Combo')) %>%
  group_by(dataset, measurement, image_type, bootstrap_id) %>%
  mutate(rank = dense_rank(-score)) %>%
  mutate(classification = ifelse(str_detect(model, 'class'), 1, 0)) %>%
  group_by(dataset, measurement, image_type, classification, rank) %>%
  summarise(rank_count = n()) %>%
  filter(rank <= 2) %>% arrange(measurement)

### Results 3.5 Depth versus Score -------------------------------------------------

# depth of the most predictive feature spaces
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(image_type, dataset, measurement) %>%
  summarise(ci = list(mean_cl_normal(model_layer_depth))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% 
  filter(image_type == 'Combo')

# linear models: score as predicted by depth
model_layer_lms <- reg_results %>% 
  group_by(model, train_type, dataset, measurement, image_type) %>%
  mutate_at(vars(score, model_layer_depth), list(z = scale)) %>%
  do(lm_fit = tidy(lm(score ~ model_layer_depth, data = .)),
     lm_fit_z = tidy(lm(score_z ~ model_layer_depth_z, data = .)))

model_layer_lms %>% unnest(lm_fit) %>% 
  filter(term == 'model_layer_depth') %>%
  group_by(train_type, dataset, measurement, image_type) %>%
  summarise(ci = list(mean_cl_normal(estimate))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% 
  filter(image_type == 'Combo', train_type == 'imagenet')

model_layer_lms %>% unnest(lm_fit) %>% 
  filter(term == 'model_layer_depth') %>%
  group_by(train_type, image_type) %>%
  summarise(ci = list(mean_cl_normal(estimate))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% 
  filter(image_type == 'Combo', train_type == 'imagenet')

model_layer_lms %>% unnest(lm_fit_z) %>% 
  filter(term == 'model_layer_depth_z') %>%
  group_by(train_type, dataset, measurement, image_type) %>%
  summarise(ci = list(mean_cl_normal(estimate))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% 
  filter(image_type == 'Combo', train_type == 'imagenet')

# average depths of maximally predictive layers by task
reg_results_max %>% filter(train_type == 'taskonomy') %>%
  group_by(model, train_type) %>%
  summarise(model_layer_depth = mean(model_layer_depth)) %>%
  arrange(desc(model_layer_depth)) %>% ungroup() %>% print(n=25)

# score ~ depth for taskonomy models
model_layer_lms %>%  unnest(lm_fit) %>% 
  filter(term == 'model_layer_depth') %>%
  select(model, train_type, dataset, 
         measurement, image_type, estimate) %>%
  group_by(model, train_type) %>% 
  summarise(estimate = list(mean_cl_normal(estimate))) %>%
  unnest(estimate) %>% filter(train_type == 'taskonomy') %>% arrange(desc(y))

### Results 3.6 Self-Supervision -------------------------------------------------

# filter for models that share architecture across train_type
shared_model_regex <- 'xcit|vit|resnet50|ResNet50-'

# filter for self-supervised models that aren't contrastive
noncontrastive_regex <- 'Cluster|Jigsaw|RotNet|PIRL'

# average scores for imagenet-trained models
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score)) %>% arrange(desc(score))

# average scores for self-supervised models
reg_results_max %>% filter(train_type == 'selfsupervised') %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score)) %>% arrange(desc(score))

# imagenet and self-supervised models ranked
reg_results_max %>% filter(train_type %in% c('imagenet', 'selfsupervised')) %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score)) %>%
  group_by(train_type) %>%
  summarise(ci = list(median_cl_boot(score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% arrange(-y)

# visualizing difference between imagenet and self-supervised models
reg_results_max %>% filter(train_type %in% c('imagenet', 'selfsupervised')) %>%
  filter(!str_detect(model, noncontrastive_regex)) %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  ggbetweenstats(x = 'train_type', y = 'score', type = 'np')

# visualizing difference between imagenet and self-supervised models across affect
reg_results_max %>% filter(train_type %in% c('imagenet', 'selfsupervised')) %>%
  filter(!str_detect(model, noncontrastive_regex)) %>%
  group_by(model, train_type, dataset) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  grouped_ggbetweenstats(x = 'train_type', y = 'score', type = 'np',
                         grouping.var = 'dataset')

reg_results_max %>% filter(train_type %in% c('selfsupervised')) %>%
  select(model, dataset) %>% distinct() %>% group_by(model) %>% count()

# rankings of models sharing architecture
reg_results_max %>% filter(!train_type %in% c('random')) %>%
  filter(str_detect(model, shared_model_regex)) %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score)) %>% 
  arrange(desc(score)) %>% print(n = 30)

# visualization of difference in models sharing architectures
reg_results_max %>% filter(!train_type %in% c('random')) %>%
  filter(str_detect(model, shared_model_regex)) %>%
  #filter(!str_detect(model, noncontrastive_regex)) %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score)) %>% ungroup() %>%
  ggbetweenstats(x = 'train_type', y = 'score', type = 'np')

# all models, including SEER and Imagenet21k trained models
# performance in pearson r
reg_results_max %>% group_by(model, train_type) %>%
  filter(train_type %in% c('imagenet21k', 'seer')) %>%
  summarise(n = n(), score_ci = mean_cl_boot(score)) %>% 
  ungroup() %>% unnest(score_ci) %>% filter(n == 21) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  mutate(rank = dense_rank(-y)) %>% arrange(rank) %>%
  filter(row_number() %% 1 == 0) %>% print(n = 250)

# all models, including SEER and Imagenet21k trained models
# performance in explained variance explained
reg_results_max %>% group_by(model, train_type) %>%
  filter(train_type %in% c('imagenet21k', 'seer')) %>%
  summarise(n = n(), score_ci = mean_cl_boot(score)) %>% 
  ungroup() %>% unnest(score_ci) %>% filter(n == 21) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  mutate(rank = dense_rank(-y)) %>% arrange(rank) %>%
  filter(row_number() %% 1 == 0) %>% print(n = 250)



### Results 3.7 Affect Comparisons -------------------------------------------------

# Aesthetics is the most predictable of the affective measures.
reg_results_max %>% filter(train_type %in% c('imagenet')) %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  ggbetweenstats(x = measurement, y = score, type = 'parametric')

reg_results_max %>% filter(train_type %in% c('imagenet')) %>%
  mutate(explained_variance = (score**2 / r**2)) %>% 
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  ggbetweenstats(x = measurement, y = score, type = 'parametric')

# pairwise comparisons between affects
reg_results_max %>% left_join(splithalf_summary) %>%
  filter(dataset == 'Oasis', image_type == 'Combo',
         train_type == 'imagenet') %>% ungroup() %>%
  {left_join(rstatix::games_howell_test(., score ~ measurement),
             rstatix::cohens_d(., score ~ measurement, hedges.correction = TRUE))}

# ... in terms of explainable variance explained
reg_results_max %>% filter(train_type %in% c('imagenet')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>% 
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  select(model, train_type, measurement, explained_variance) %>%
  pivot_wider(names_from = measurement, values_from = explained_variance) %>%
  summarise(beauty_versus_arousal = mean_cl_boot(Beauty - Arousal),
            beauty_versus_valence = mean_cl_boot(Beauty - Valence))
            

reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>% 
  filter(dataset == 'Oasis', image_type == 'Combo',
         train_type == 'imagenet') %>% ungroup() %>%
  {left_join(rstatix::t_test(., explained_variance ~ measurement),
             rstatix::cohens_d(., explained_variance ~ measurement, hedges.correction = TRUE))}

# pairwise tests of beauty vs arousal vs valence (pearson's r)
reg_results_max %>% filter(train_type == 'imagenet') %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  ggbetweenstats(x = 'measurement', y = 'score') %>% 
  extract_stats() %$% pairwise_comparisons_data %>%
  select(group1, group2, statistic,  p.value, p.value.adjustment, method)

# pairwise tests of beauty vs arousal vs valence (variance explained)
reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>% 
  filter(dataset == 'Oasis', image_type == 'Combo',
         train_type == 'imagenet') %>% ungroup() %>%
  ggbetweenstats(x = 'measurement', y = 'explained_variance') %>% 
  extract_stats() %$% pairwise_comparisons_data %>%
  select(group1, group2, statistic,  p.value, p.value.adjustment, method)

# average advantage (in hedges' g) of beauty over arousal and valence
reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>% 
  filter(dataset == 'Oasis', train_type == 'imagenet') %>% group_by(image_type) %>%
  {left_join(rstatix::t_test(., explained_variance ~ measurement),
             rstatix::cohens_d(., explained_variance ~ measurement, hedges.correction = TRUE))} %>%
  filter(p.adj < 0.001) %>% filter(group1 == 'Beauty' | group2 == 'Beauty') %>% 
  summarise(mean_hedges_g = mean(abs(effsize)))

# correlation between ratings of beauty, arousal, and valence
oasis_responses %>% cor_mat(vars = c('beauty', 'arousal', 'valence')) %>%
  column_to_rownames('rowname')


### Results 3.8 Type Comparisons -------------------------------------------------

# Are there differences across image category?
reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(!image_type %in% c('Combo'), dataset == 'Oasis', 
         train_type %in% c('imagenet','selfsupervised')) %>%
  grouped_ggbetweenstats(x = image_type, y = explained_variance, 
                         grouping.var = measurement, type = 'parametric')

# range of scores (mean, ci) across image_type, averaging measurement
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(train_type, dataset, image_type) %>% 
  summarise(ci = list(mean_cl_normal(score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  arrange(desc(y)) #%>% filter(image_type %in% c('Scene','Person'))

# range of scores (mean, ci) across image_type, no averaging
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(dataset, measurement, image_type) %>% 
  summarise(ci = list(mean_cl_normal(score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  filter(measurement == 'Beauty') %>% arrange(desc(y)) #%>% 
  filter(image_type %in% c('Scene','Person'))

# pairwise tests between image_types, no averaging: nonsignificant
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(model, train_type, dataset, image_type, measurement) %>% 
  summarise(score = mean(score)) %>% group_by(measurement, dataset) %>%
  {left_join(rstatix::t_test(., score ~ image_type),
             rstatix::cohens_d(., score ~ image_type))} %>% filter(p > 0.001)

# pairwise test between most | least predictive image_type (oasis)
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(model, train_type, dataset, image_type) %>% 
  summarise(score = mean(score)) %>% ungroup() %>%
  {left_join(rstatix::t_test(., score ~ image_type),
             rstatix::cohens_d(., score ~ image_type))} %>%
  filter(group1 %in% c('Scene', 'Person') &
           group2 %in% c('Scene','Person'))

# pairwise test between most | least predictive image_type (vessel)
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(model, train_type, dataset, image_type) %>% 
  summarise(score = mean(score)) %>% ungroup() %>%
  {left_join(rstatix::t_test(., score ~ image_type),
             rstatix::cohens_d(., score ~ image_type))} %>%
  filter(group1 %in% c('fac', 'art') &
           group2 %in% c('fac','art'))

# pairwise_tests between image_catgories, merging datasets
reg_results_max %>% filter(train_type == 'imagenet') %>%
  group_by(model, train_type, dataset, image_type) %>% 
  summarise(score = mean(score)) %>% ungroup() %>%
  {left_join(rstatix::t_test(., score ~ image_type),
             rstatix::cohens_d(., score ~ image_type))} %>%
  filter(p.adj.signif == 'ns')

### Results 3.9 Cross-Decoding -------------------------------------------------

# load cross-decoding scores computed separately in Python
cross_decoding <- read_csv('incoming/cross_regs/image_type/swin_base_patch4_window7_224_imagenet.csv')
# cross_decoding <- read_csv('results/cross_decoding_scores.csv')

cat_names = c('Oasis - Animal', 'Oasis - Object', 'Oasis - Person', 'Oasis - Scene', 
              'Vessel - Indoor', 'Vessel - Outdoor', 'Vessel - Art', 
              'Vessel - Faces', 'Vessel - Scene', 'Oasis - Combo', 'Vessel - Combo')

cat_names = c('O:Animal', 'O:Object', 'O:Person', 'O:Scene', 
              'V:Indoor', 'V:Outdoor', 'V:Art', 'V:Faces','V:Scene', 'O:Combo', 'V:Combo')

cross_decoding %>% select(train_data, test_data, cross_score) %>% 
  set_colnames(c('var1', 'var2', 'cor') )%>% cor_spread() %>% 
  relocate(Object, .after = rowname) #%>% column_to_rownames('rowname')

cross_decoding %>% select(train_data, test_data, cross_score) %>%
  set_colnames(c('var1', 'var2', 'cor')) %>% cor_spread() %>% 
  relocate(Object, .after = rowname) %>% column_to_rownames('rowname') %>%
  set_colnames(cat_names) %>% set_rownames(cat_names) %>% cor_gather() %>%
  set_colnames(c('train_data', 'test_data', 'cross_score')) %>%
  mutate(train_data = factor(train_data, levels = cat_names),
         test_data = factor(test_data, levels = cat_names)) %>%
  #mutate_at(c('train_data', 'test_data'), str_replace, ' - ', ':') %>%
  ggplot(aes(test_data, train_data, fill=cross_score, 
             label=round(cross_score, 2))) + geom_tile() +
  labs(x = NULL, y = NULL, fill = "Cross-Decoding\nScore") +
  scale_fill_gradient2(low="#A63446", mid = 'white', high="#0C6291", limits=c(-1,1)) +
  geom_text(size = 7) + theme_minimal() +
  scale_x_discrete(expand=c(0,0), position = 'top') +
  scale_y_discrete(expand=c(0,0), limits = rev(cat_names)) +
  #theme(text=element_text(family="Roboto")) + 
  easy_rotate_x_labels(angle = 25, side = 'left') +
  theme(text=element_text(size = 24))

cross_decoding %>% select(train_data, test_data, cross_score) %>%
  filter(train_data != test_data) %>%
  filter(!str_detect(train_data, 'combo'),
         !str_detect(test_data, 'combo')) %>% 
  group_by(train_data) %>% 
  summarise(ci = list(mean_cl_boot(cross_score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% arrange(-y)

cross_decoding_overall <- dir('incoming/cross_regs/image_type', pattern = '.csv', full.names = TRUE, recursive = TRUE) %>%
  map(read_csv_add_dataset) %>% bind_rows() %>% filter(model != 'alexnet')

cross_decoding_overall %>% select(model, train_type, train_data, test_data, cross_score) %>%
  mutate(train_data = fct_recode(train_data, !!!image_type_labels[-10]),
         test_data = fct_recode(test_data, !!!image_type_labels[-10])) %>%
  filter(train_data != test_data) %>%
  filter(!str_detect(train_data, 'combo'),
         !str_detect(test_data, 'combo')) %>% 
  group_by(model, train_type, train_data) %>% 
  summarise(cross_score = mean(cross_score)) %>%
  ungroup() %>% group_by(train_data) %>% 
  summarise(n = n(), ci = list(mean_cl_boot(cross_score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>% arrange(-y)
  

### Results 3.10 Hipster Detection ---------------------------------

read_csv_subjects <- function(x) {read_csv(x, col_types = cols()) %>% 
    mutate(dataset = 'oasis')}

# import individual subject regression scores per model
subject_regs <- dir('incoming/subject_regs', pattern = '.csv', full.names = TRUE, recursive = TRUE) %>%
  map(read_csv_subjects) %>% bind_rows() %>% fix_factor_levels() %>% 
  filter(model != 'alexnet', score_type == 'pearson_r') %>%
  mutate(train_type = str_replace(train_type, detectron_regex, 'detectron'))

# average scores per model across subjects
subject_regs %>% group_by(model, train_type, dataset, measurement, image_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  group_by(dataset, measurement, image_type) %>%
  summarise(score_ci = mean_cl_boot(score)) %>%
  unnest(score_ci) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  filter(dataset == 'Oasis', image_type == 'Combo')

# average scores for individual subjects, best and worst
subject_regs %>% group_by(subject, dataset, measurement, image_type) %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  summarise(score_ci = mean_cl_boot(score)) %>% 
  unnest(score_ci) %>% arrange(desc(y)) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  group_by(dataset, measurement) %>%
  filter(y == min(y, na.rm = TRUE) | 
         y == max(y, na.rm = TRUE)) %>%
  arrange(dataset, measurement, image_type)

# the most predictive models for individual subjects
subject_regs %>% group_by(model, train_type, dataset, measurement, image_type) %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  summarise(score = mean_cl_normal(score, na.rm = TRUE)) %>% 
  filter(measurement == 'Beauty') %>% arrange(-score$y)
  

# combining individual subject scores with mmo correlations
mmo_reg_combo <- subject_regs %>% select(-image_count) %>% right_join(mmo_corrs) 

mmo_reg_corrs <- mmo_reg_combo %>% 
  group_by(model, train_type, dataset, 
           measurement, image_type) %>% cor_test(score, corr)

# mmo-score corrs for combined images
mmo_reg_corrs %>% filter(p < 0.5) %>%
  filter(image_type == 'Combo') %>%
  group_by(dataset, measurement, image_type) %>%
  summarise(n = n(), cor_ci = mean_cl_boot(cor)) %>%
  unnest(cor_ci) %>% 
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']'))

# mmo-score corrs for other categories
mmo_reg_corrs %>% filter(p < 0.5) %>% 
  filter(image_type != 'Combo') %>%
  group_by(dataset, measurement, image_type) %>%
  summarise(n = n(), cor_ci = mean_cl_boot(cor)) %>%
  unnest(cor_ci) %>% 
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']'))

# plot of mmo corr versus score across models
mmo_reg_corrs %>% #filter(p < 0.5) %>% 
  filter(dataset == 'Oasis', image_type != 'Combo') %>%
  grouped_ggbetweenstats(x = 'image_type', y = 'cor', grouping.var = 'measurement',
                         results.subtitle = FALSE, pairwise.comparisons = FALSE,
                         ggplot.component = scale_y_continuous(limits = c(-0.25, 0.6)))

mmo_reg_corrs %>% filter(p < 0.5) %>% 
  filter(dataset == 'vessel', image_type != 'Combo') %>%
  grouped_ggbetweenstats(x = 'image_type', y = 'cor', grouping.var = 'measurement',
                         results.subtitle = FALSE, pairwise.comparisons = FALSE,
                         ggplot.component = scale_y_continuous(limits = c(-0.5, 1.0)))

mmo_reg_combo %>% group_by(subject, measurement, image_type, dataset) %>%
  summarise(n = n(), score = mean(score, na.rm = TRUE), 
            mmo_corr = mean(corr, na.rm = TRUE)) %>%
  ungroup() %>% filter(dataset == 'Oasis', image_type == 'Combo') %>%
  grouped_ggscatterstats(x = score, y = mmo_corr, grouping.var = measurement,
                         ggtheme = theme_bw())

mmo_reg_combo %>% group_by(subject, measurement, image_type, dataset) %>%
  summarise(n = n(), score = mean(score, na.rm = TRUE), 
            mmo_corr = mean(corr, na.rm = TRUE)) %>%
  ungroup() %>% filter(dataset == 'Oasis', image_type == 'Combo') %>%
  ggplot(aes(x = score, y = mmo_corr, group = measurement)) +
  facet_wrap(~measurement) + geom_point() + geom_smooth(method = 'lm')

### Results 3.10 CLIP + Language ---------------------------------

shared_model_regex <- 'vit|ViT|resnet50|RN50|resnet101|RN101'

# bootstrapped rankings of models, averaged
boot_results %>% filter(image_type == 'Combo') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance, na.rm = TRUE)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = dense_rank(explained_variance)) %>% arrange(rank) %>%
  mutate_at(vars(explained_variance, lower_ci, upper_ci), round, 3) %>%
  mutate(report = paste0(explained_variance,' [',lower_ci,', ',upper_ci,']')) %>%
  select(model, train_type, report, rank) %>% print(n = 200)    

#bootstapped rankings of models, by image category
boot_results %>% filter(image_type == 'Art') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, dataset, measurement, image_type, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance, na.rm = TRUE)) %>%
  group_by(model, train_type, dataset, measurement, image_type) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = dense_rank(-explained_variance)) %>% arrange(rank) %>%
  mutate_at(vars(explained_variance, lower_ci, upper_ci), round, 3) %>%
  mutate(report = paste0(explained_variance,' [',lower_ci,', ',upper_ci,']')) %>%
  select(model, train_type, dataset, measurement, image_type, report, rank) %>% print(n = 10)    

reg_results_max %>% filter(model == 'vit_base_patch16_224' | model == 'ViT-B/16') %>%
  left_join(splithalf_summary) %>%
  mutate(train_type = str_to_title(train_type)) %>%
  mutate(score = ifelse(score < 0, 0, score)) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  ggwithinstats(x = 'train_type', y = 'explained_variance',
                 results.subtitle = FALSE, type = 'p')

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  #filter(str_detect(model, shared_model_regex)) %>%
  select(model, train_type, dataset, measurement, 
         image_type, explained_variance) %>%
  arrange(desc(explained_variance))

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  mutate(language = ifelse(str_detect(train_type, 'clip|slip'), 'clip','nonclip')) %>%
  select(model, train_type, dataset, measurement, 
         image_type, language, explained_variance) %>%
  group_by(dataset, measurement, image_type, language) %>%
  mutate(rank = dense_rank(-explained_variance)) %>%
  filter(rank == 1) %>% arrange(dataset, measurement, image_type, language) %>% print(n = 50)

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  #filter(str_detect(model, shared_model_regex)) %>%
  select(model, train_type, dataset, measurement, 
         image_type, explained_variance) %>%
  group_by(dataset, measurement, image_type) %>%
  mutate(rank = dense_rank(-explained_variance)) %>%
  arrange(dataset, measurement, image_type, rank) %>%
  filter(rank <= 2) %>% print(n = 50)

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  #filter(str_detect(model, shared_model_regex)) %>%
  select(model, train_type, dataset, measurement, 
         image_type, explained_variance) %>%
  group_by(dataset, measurement, image_type) %>% 
  mutate(rank = dense_rank(-explained_variance)) %>% 
  filter(image_type == 'art') %>% arrange(rank)

boot_results %>% filter(image_type == 'Combo') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  select(model, train_type, dataset, measurement, image_type, 
         bootstrap_id, score, explained_variance) %>%
  filter((model == 'ViT-L/14' & train_type == 'clip') |
           (model == 'convnext_large_in22k' & train_type == 'imagenet21k')) %>% 
  mutate(model = ifelse(model == 'ViT-L/14', 'clip', 'nonclip')) %>%
  select(-train_type, -score) %>% arrange(desc(model), bootstrap_id) %>%
  pivot_wider(names_from = model, values_from = explained_variance) %>%
  mutate(clip_difference = clip - nonclip) %>%
  mutate(clip_better = clip_difference > 0) %>%
  group_by(dataset, measurement, image_type) %>%
  summarise(clip_better = sum(clip_better))

boot_results %>% filter(image_type == 'Combo') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  select(model, train_type, dataset, measurement, image_type, 
         bootstrap_id, score, explained_variance) %>%
  filter((model == 'ViT-L/14' & train_type == 'clip') |
           (model == 'ViT-L-SLIP' & train_type == 'slip')) %>% 
  group_by(model, train_type, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance)) %>% ungroup() %>%
  mutate(model = ifelse(model == 'ViT-L/14', 'clip', 'slip')) %>%
  select(-train_type) %>% arrange(desc(model), bootstrap_id) %>%
  pivot_wider(names_from = model, values_from = explained_variance) %>%
  mutate(clip_difference = clip - slip) %>%
  mutate(clip_better = clip_difference > 0) %>%
  #summarise(clip_better = sum(clip_better)) %>%
  summarise(lower_ci = quantile(clip_difference, 0.025, na.rm = TRUE),
            upper_ci = quantile(clip_difference, 0.975, na.rm = TRUE),
            difference = mean(clip_difference, na.rm = TRUE)) %>% ungroup()

reg_results_max %>% filter(train_type == 'slip') %>%
  select(model, train_type) %>% distinct()

pairwise_subset <- reg_results_max %>% filter(image_type == 'Combo') %>%
  filter(train_type == 'slip', !str_detect(model, 'Ep100|CC12M')) %>%
  mutate(slip_kind = str_split(model, '-', simplify = TRUE)[, 3]) %>%
  mutate(slip_size = str_split(model, '-', simplify = TRUE)[, 2],
         slip_size = fct_recode(factor(slip_size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(slip_kind, slip_size) %>%
  summarise(explained_variance = mean(explained_variance))

inner_join(pairwise_subset, pairwise_subset, by = 'slip_kind') %>%
  rowwise() %>%
  mutate(mirrors = paste(sort(c(slip_size.x, slip_size.y)), collapse = "-")) %>%
  group_by(slip_kind, mirrors) %>%
  slice(1) %>%
  ungroup() %>%
  filter(slip_size.x != slip_size.y) %>%
  select(-mirrors)

pairwise_subset <- boot_results %>% filter(image_type == 'Combo') %>%
  filter(train_type == 'slip', !str_detect(model, 'Ep100|CC12M')) %>%
  mutate(slip_kind = str_split(model, '-', simplify = TRUE)[, 3]) %>%
  mutate(slip_size = str_split(model, '-', simplify = TRUE)[, 2],
         slip_size = fct_recode(factor(slip_size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(slip_kind, slip_size, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance))

inner_join(pairwise_subset, pairwise_subset, by = c('slip_size', 'bootstrap_id')) %>%
  rowwise() %>%
  mutate(mirrors = paste(sort(c(slip_kind.x, slip_kind.y)), collapse = "-")) %>%
  group_by(bootstrap_id, slip_size, mirrors) %>%
  slice(1) %>%
  ungroup() %>%
  filter(slip_kind.x != slip_kind.y) %>%
  select(-mirrors) %>%
  mutate(difference = explained_variance.x - explained_variance.y) %>%
  select(-explained_variance.x, -explained_variance.y) %>%
  group_by(slip_size, slip_kind.x, slip_kind.y) %>%
  summarise(lower_ci = quantile(difference, 0.025, na.rm = TRUE),
            upper_ci = quantile(difference, 0.975, na.rm = TRUE),
            difference = mean(difference, na.rm = TRUE)) %>% ungroup()

inner_join(pairwise_subset, pairwise_subset, by = c('slip_size', 'bootstrap_id')) %>%
  rowwise() %>%
  mutate(mirrors = paste(sort(c(slip_kind.x, slip_kind.y)), collapse = "-")) %>%
  group_by(bootstrap_id, slip_size, mirrors) %>%
  slice(1) %>%
  ungroup() %>%
  filter(slip_kind.x != slip_kind.y) %>%
  select(-mirrors) %>%
  mutate(diff_direction = ifelse(explained_variance.x > explained_variance.y, 'positive', 'negative')) %>%
  select(-explained_variance.x, -explained_variance.y) %>%
  group_by(slip_size, slip_kind.x, slip_kind.y, diff_direction) %>%
  summarise(count = n())

boot_results %>% filter(image_type == 'Combo') %>%
  filter(train_type == 'slip', !str_detect(model, 'Ep100|CC12M')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  select(model, train_type, dataset, measurement, image_type, 
         bootstrap_id, score, explained_variance) %>%
  select(-train_type, -score) %>% arrange(desc(model), bootstrap_id)

### Appendix: Model Metadata -----------------------------------------------

reg_results %>% left_join(model_typology %>% select(model, train_type, model_class)) %>%
  group_by(model, train_type, dataset, measurement, image_type) %>%
  filter(!str_detect(model_layer, 'Dropout')) %>%
  filter(score == max(score)) %>% ungroup() %>% 
  arrange(desc(score)) %>% select(-alpha, -score_type) %>%
  filter(measurement == 'Beauty', image_type == 'Combo',
         train_type == 'imagenet', !str_detect(model_class, 'MLP-Mixer')) %>% 
  grouped_ggbetweenstats(x = model_class, y = score, grouping.var = dataset, type = 'np')

reg_results %>% left_join(model_typology %>% select(model, train_type, model_class)) %>%
  filter(model_depth < 500) %>%
  group_by(model, train_type, dataset, measurement, image_type) %>%
  filter(!str_detect(model_layer, 'Dropout')) %>%
  filter(score == max(score)) %>% ungroup() %>% 
  arrange(desc(score)) %>% select(-alpha, -score_type) %>%
  filter(measurement == 'Beauty', image_type == 'Combo',
         train_type %in% c('imagenet')) %>%
  grouped_ggscatterstats(x = model_depth, y = score, grouping.var = 'dataset', type = 'robust')

