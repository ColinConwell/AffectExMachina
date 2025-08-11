setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('psych', 'mgsub', 'arrow', 'scales', 'weights', 'mgcv', 'skimr',
               'ggeasy', 'ggpubr', 'ggfortify', 'ggstatsplot', 'cowplot', 
               'stargazer', 'rstatix', 'tidymodels', 'tidytext', 'tidyverse')

#https://rpubs.com/dgolicher/median_boot
ci_norm <- function(x) {return(qnorm(0.975) * (sd(x) / sqrt(n())))}
ci_boot <- function(x) {return(mean_cl_boot(x) %>% mutate(ci=ymax-y) %>% pull(ci))}

custom_themes <- list()
theme_set(theme_bw())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('psych', 'mgsub', 'arrow', 'scales', 'weights', 'mgcv', 'skimr',
               'ggeasy', 'ggpubr', 'ggfortify', 'ggstatsplot', 'cowplot', 
               'stargazer', 'rstatix', 'tidymodels', 'tidytext', 'tidyverse')

#https://rpubs.com/dgolicher/median_boot
ci_norm <- function(x) {return(qnorm(0.975) * (sd(x) / sqrt(n())))}
ci_boot <- function(x) {return(mean_cl_boot(x) %>% mutate(ci=ymax-y) %>% pull(ci))}

custom_themes <- list()
theme_set(theme_bw())

##### Specifications ------------------------------------------------------

custom_themes[['bottom_legend']] <- theme(legend.position="bottom", legend.justification="center", 
                                          legend.box.margin=margin(-12,0,0,0))

##### Metric Correlations ------------------------------------------------

metric_corr_results <- read_csv('results/metric_correlations.csv') %>%
  mutate(corr_sq = corr**2, corr_abs = abs(corr)) %>%
  filter(duplicated(corr_abs) == FALSE) %>%
  filter(model != 'googlenet')

devtools::install_github("teunbrand/ggh4x")
library('ggh4x')

metric_titles <- c(MeanActivity = 'mean_activity', Sparseness = 'sparseness')
train_type_titles <- c(ImageNet = 'imagenet', Random = 'random')
metric_corr_results %>% filter(dataset == 'oasis') %>%
  mutate(metric = fct_recode(as.factor(metric), !!!metric_titles)) %>%
  group_by(model, train_type, image_type, measurement, metric) %>%
  filter(!model %in% metric_corr_error_models) %>%
  filter(train_type != 'taskonomy') %>% 
  filter(corr_abs == max(corr_abs, na.rm = TRUE)) %>%
  group_by(model, metric, measurement, image_type) %>%
  mutate(slope = (corr_abs[train_type=='imagenet'] - corr_abs[train_type=='random'])/(2-1)) %>%
  mutate(train_type = fct_recode(as.factor(train_type), !!!train_type_titles)) %>%
  mutate(measurement = str_to_title(measurement)) %>%
  ggplot(aes(train_type, corr_abs)) + 
  facet_nested(measurement ~ image_type + metric) +
  geom_boxplot(width = 0.15) + geom_point() + 
  geom_line(aes(group=model, col = slope > 0)) +
  ylab('Score') + xlab('Weights') + guides(color = FALSE)

metric_corr_results %>% filter(dataset == 'vessel') %>%
  mutate(metric = fct_recode(as.factor(metric), !!!metric_titles)) %>%
  group_by(model, train_type, image_type, measurement, metric) %>%
  filter(train_type != 'taskonomy') %>% 
  filter(corr_abs == max(corr_abs, na.rm = TRUE)) %>%
  group_by(model, metric, measurement, image_type) %>%
  mutate(slope = (corr_abs[train_type=='imagenet'] - corr_abs[train_type=='random'])/(2-1)) %>% 
  filter(slope < 0 & train_type == 'imagenet') %>% 
  select(model, train_type, dataset, image_type, 
         measurement, metric, corr_abs, slope) %>% 
  filter(train_type == 'imagenet') %>% filter(grepl('it|mix|swin', model))

metric_corr_results %>% filter(dataset == 'vessel') %>%
  mutate(metric = fct_recode(as.factor(metric), !!!metric_titles)) %>%
  group_by(model, train_type, image_type, measurement, metric) %>%
  filter(train_type != 'taskonomy') %>% 
  filter(corr_abs == max(corr_abs, na.rm = TRUE)) %>%
  group_by(model, metric, measurement, image_type) %>%
  mutate(slope = (corr_abs[train_type=='imagenet'] - corr_abs[train_type=='random'])/(2-1)) %>%
  mutate(train_type = fct_recode(as.factor(train_type), !!!train_type_titles)) %>%
  mutate(measurement = str_to_title(measurement)) %>%
  ggplot(aes(train_type, corr_abs)) + 
  facet_nested(measurement ~ image_type + metric) +
  geom_boxplot(width = 0.15) + geom_point() + 
  geom_line(aes(group=model, col = slope > 0)) +
  ylab('Score') + xlab('Weights') + guides(color = FALSE)

metric_corr_results %>% filter(train_type != 'taskonomy') %>%
  filter(!model %in% metric_corr_error_models) %>%
  group_by(model, train_type, measurement, image_type, metric) %>%
  filter(corr_sq == max(corr_sq, na.rm = TRUE)) %>% 
  group_by(dataset, image_type, measurement, metric) %>%
  {left_join(rstatix::t_test(., corr_sq ~ train_type, paired = TRUE) %>% 
               adjust_pvalue(method = 'bonferroni') %>% add_significance('p.adj'),
             cohens_d(., corr_sq ~ train_type, paired = TRUE))} %>% 
  select(image_type, measurement, metric, group1, group2, 
         p.adj, p.adj.signif, effsize, magnitude) %>% filter(p.adj > 0.01 & effsize < 0)

metric_corr_results %>% group_by(model, train_type, dataset, image_type, metric, measurement) %>%
  filter(corr_abs == max(corr_abs)) %>% group_by(model, train_type, metric) %>%
  summarise(corr_abs = mean(corr_abs)) %>% arrange(metric, desc(corr_abs))

##### Metric Permutations -----------------------------------------------

metric_permutes <- read_csv('results/metric_permuting.csv') %>%
  mutate(dataset = ifelse(image_type %in% oasis_levels, 'oasis', 'vessel')) %>%
  mutate(pass_permutation = as.factor(ifelse(corr_p_value < 0.05, 'Yes', 'No')))

metric_permutes %>% filter(corr_p_value < 0.05) %>%
  filter(dataset == 'oasis') %>% group_by(measurement) %>% tally()

metric_permutes %>% filter(corr_p_value < 0.05) %>%
  filter(dataset == 'oasis') %>% group_by(measurement, image_type, metric) %>%
  summarise(corr_max_score = mean(corr_max_score)) %>% group_by(metric) %>%
  summarise(ci = ci_boot(corr_max_score), n = n(), 
            corr_max_score = mean(corr_max_score))

metric_permutes %>% filter(corr_p_value < 0.05) %>%
  filter(dataset == 'oasis') %>% group_by(measurement, image_type) %>%
  summarise(ci = ci_boot(corr_max_score), n = n(),
            corr_max_score = mean(corr_max_score)) %>% arrange(corr_max_score)

metric_permutes %>% filter(corr_p_value < 0.05) %>%
  filter(dataset == 'oasis') %>% group_by(measurement, image_type) %>%
  summarise(ci = ci_boot(corr_max_score), 
            corr_max_score = mean(corr_max_score)) %>% 
  group_by(measurement) %>% summarise(ci = ci_boot(corr_max_score), n = n(),
                                      corr_max_score = mean(corr_max_score))

metric_permutes %>% filter(corr_p_value < 0.05) %>%
  filter(dataset == 'oasis') %>% group_by(measurement) %>%
  summarise(ci = ci_boot(corr_max_score), n = n(),
            corr_max_score = mean(corr_max_score))

metric_permutes %>% filter(corr_p_value < 0.05) %>% 
  filter(dataset == 'oasis') %>% group_by(model, train_type, model_depth) %>%
  summarise(corr_max_score = mean(corr_max_score)) %>% 
  filter(train_type != 'taskonomy') %>% group_by(train_type) %>% 
  cor_test(model_depth, corr_max_score, method = 'pearson')

metric_corr_permutes <- metric_corr_results %>%
  group_by(model, train_type, image_type, measurement, metric) %>%
  filter(corr_abs == max(corr_abs, na.rm = TRUE)) %>% ungroup %>%
  left_join(metric_permutes %>% select(model, train_type, image_type,
                                       measurement, metric, pass_permutation))

devtools::install_github("eliocamp/ggnewscale")
library('ggnewscale')

choice_palette <- c(
  `blue`        = "#00BFC4",
  `black`      = "black")

metric_corr_permutes %>% filter(train_type != 'taskonomy') %>%
  filter(dataset == 'oasis' & image_type != 'Combo') %>%
  group_by(model, metric, measurement, image_type) %>%
  mutate(slope = (corr[train_type=='imagenet'] - 
                    corr[train_type=='random'])/(2-1)) %>%
  mutate(imagenet_better = case_when(
    corr >= 0 & slope >= 0 ~ 'Yes', 
    corr < 0 & slope >= 0 ~ 'No',
    corr >= 0 & slope <= 0 ~ 'No', 
    corr < 0 & slope <= 0 ~ 'Yes', 
    TRUE ~ 'No')) %>%
  ggplot(aes(train_type, corr)) + 
  facet_nested(measurement ~ image_type + metric) +
  geom_line(aes(group=model, col = imagenet_better)) +
  #geom_line(aes(group=model, col = slope > 0)) +
  scale_color_manual(element_blank(), values = c('black', '#00BFC4'), breaks = TRUE,
                     labels = c(NULL, 'ImageNet-Trained Max Score Higher than Randomly-Initialized')) +
  new_scale_color() + 
  geom_point(aes(color = pass_permutation), size = 1.5) +
  #geom_boxplot(width = 0.15, aes(color = pass_permutation)) + 
  scale_color_manual(element_blank(), values = c('black','violet'), breaks = 1,
                     labels = c(NULL, 'Max Score Passes a Permutation Test at p < 0.01')) +
  labs(x = element_blank(), y = 'Score (Pearson R)') + custom_themes[['bottom_legend']]

metric_corr_permutes %>% filter(train_type != 'taskonomy') %>%
  filter(dataset == 'oasis' & image_type != 'Combo') %>%
  group_by(model, metric, measurement, image_type) %>%
  mutate(slope = (corr_abs[train_type=='imagenet'] - 
                    corr_abs[train_type=='random'])/(2-1)) %>%
  ggplot(aes(train_type, corr_abs)) + 
  facet_nested(measurement ~ image_type + metric) +
  geom_boxplot(width = 0.15) + 
  geom_point(aes(color = pass_permutation), show.legend = FALSE) +
  scale_color_manual('Pass Permutation', values = c('black','violet')) +
  geom_line(aes(group=model, linetype = slope < 0), show.legend = FALSE) +
  ylab('Score') + xlab('Weights')

metric_corr_permutes %>% filter(train_type != 'taskonomy') %>%
  filter(dataset == 'oasis' & image_type != 'Combo') %>%
  group_by(model, metric, measurement, image_type) %>%
  mutate(slope = (corr_abs[train_type=='imagenet'] - corr_abs[train_type=='random'])/(2-1)) %>%
  ggplot(aes(train_type, corr_abs)) + 
  facet_nested(measurement ~ image_type + metric) +
  geom_boxplot(width = 0.15, outlier.shape = NA) + 
  geom_point(aes(alpha = pass_permutation), size = 2, show.legend = FALSE) +
  scale_alpha_manual('Pass Permutation', values = c(0.1,1)) +
  geom_line(aes(group=model, color = slope > 0), show.legend = FALSE) +
  ylab('Score') + xlab('Weights')

# default colors: #F8766D, #00BFC4

##### Metric Analysis -----------------------------------------------------

metric_analysis <- read_csv('results/metric_analysis.csv')

metric_analysis %>% mutate(score = ridge_gcv_score) %>% 
  filter(train_type == 'taskonomy') %>%
  filter(image_type %in% c('lsc')) %>%
  filter(!str_detect(model,'random_weights')) %>%
  mutate(rank = sprintf("%02i", as.integer(rank(score)))) %>% 
  {ggplot(., aes(rank, score)) +
      geom_bar(stat = 'identity', position = 'identity') + 
      xlab('Model') + ylab('Score (Pearson R)') + labs(fill = 'Task Cluster') +
      scale_x_discrete(labels = with(., model %>% set_names(rank))) +
      facet_wrap(~interaction(metric, image_type, sep = ' | '), scales = 'free') +
      geom_label(aes(y = 0.7, label = round(score, 3)), show.legend = FALSE) +
      theme(legend.position="bottom",
            legend.justification="center", 
            legend.box.margin=margin(-12,0,0,0)) +
      coord_flip(ylim = c(0,0.75), clip = 'on') + theme_bw()}

##### Metric Stepwise -----------------------------------------

metric_stepwise <- read_csv('results/stepwise_regressions2.csv')

test_models <- c('alexnet','vgg19','resnet18', 'densenet121','resnet101','resnet152')
metric_stepwise %>% filter(model %in% test_models) %>%
  ggplot(aes(model_layer_depth, score, color = model, linetype = train_type)) + 
  facet_grid(metric ~ image_type) + geom_line() + ylim(c(-0.1, 1.0))

metric_stepwise %>% filter(train_type != 'taskonomy') %>%
  filter(model_layer_depth == 1.0) %>%
  ggplot(aes(model_depth, score, color = metric)) + 
  facet_grid(train_type ~ image_type) + geom_point() + 
  geom_smooth(method = 'lm') + ylim(c(-0.1, 1.0))