setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('psych', 'mgsub', 'arrow', 'scales', 'weights', 'mgcv', 'skimr',
               'ggeasy', 'ggpubr', 'ggfortify', 'ggstatsplot', 'cowplot', 
               'stargazer', 'rstatix', 'tidymodels', 'tidytext', 'tidyverse')

ci_norm <- function(x) {return(qnorm(0.975) * (sd(x) / sqrt(n())))}
ci_boot <- function(x) {return(mean_cl_boot(x) %>% mutate(ci=ymax-y) %>% pull(ci))}

custom_themes <- list()
theme_set(theme_bw())

#####---------------------------------------------------------------------

results_raw_wide <- read.csv('raw_results.csv')
results_analysis <- read.csv('analysis_results.csv')
results_corr_long <- read.csv('correlation_results.csv') %>% 
  group_by(model, image_type, metric) %>% 
  mutate(model_depth = n()) %>% ungroup() %>% 
  mutate(model_layer_relative = model_layer_index / model_depth)

results_corr_long %>% filter(metric %in% c('mean_activity', 'sparseness')) %>%
  ggplot(aes(model_layer_relative, corr, color = model)) + 
  facet_grid(cols = vars(image_type), rows=vars(metric)) +
  geom_smooth(method = 'gam', se = FALSE) + theme_bw()

results_corr_long %>% filter(metric %in% c('mean_activity', 'sparseness')) %>%
  mutate(model_kind = word(str_replace(model, '[:digit:]', '_'), sep = fixed('_'))) %>%
  ggplot(aes(model_layer_relative, corr, color = model)) + 
  facet_grid(cols = vars(image_type), rows=vars(metric)) +
  geom_smooth(method = 'lm', se = FALSE) + theme_bw() +
  labs(x = 'Correlation', y = 'Model Layer Depth (Relative)')

results_corr_long %>% filter(metric %in% c('mean_activity', 'sparseness')) %>%
  mutate(model_kind = word(str_replace(model, '[:digit:]', '_'), sep = fixed('_'))) %>%
  ggplot(aes(model_layer_relative, corr, group=model, color = model_kind)) + 
  facet_grid(cols = vars(image_type), rows=vars(metric)) +
  geom_smooth(method = 'lm', se = FALSE) + theme_bw() +
  labs(y = 'Correlation', x = 'Model Layer Depth (Relative)')

results_corr_max <- results_corr_long %>% 
  group_by(model, metric, image_type) %>%
  filter(corr == max(corr) | corr == min(corr)) %>%
  mutate(corr_pole = case_when(corr == min(corr) ~ 'min', 
                               corr == max(corr) ~ 'max', TRUE ~ 'other')) %>%
  filter(corr_pole != 'other') %>% select(-corr, corr) %>%
  mutate(corr_sign = ifelse(corr < 0, '-', '+')) %>%
  mutate(corr_abs = abs(corr)) %>% ungroup()

view(results_corr_max %>% group_by(metric, image_type, corr_pole) %>% summarise(mean = mean(corr)))

results_corr_max %>% 
  filter(metric %in% c('mean_activity','sparseness')) %>%
ggplot(aes(corr_pole, corr_abs)) + 
  facet_grid(cols = vars(image_type), row = vars(metric)) +
  geom_boxplot(width = 0.15) + geom_line(aes(group = model), alpha = 0.5) + geom_point(aes(color = corr_sign))
  

#####---------------------------------------------------------------------

results_corr_long %>% #filter(metric == 'mean_activity' & image_type == 'lsc') %>%
  filter(metric %in% c('mean_activity', 'sparseness')) %>%
  mutate(model_kind = word(str_replace(model, '[:digit:]', '_'), sep = fixed('_'))) %>%
  mutate(model_type = ifelse(model_kind == 'vgg' | model_kind == 'alexnet' | model_kind == 'resnet', 
         'Alexnet|VGG|ResNet', 'EverythingElse')) %>%
  ggplot(aes(model_layer_relative, corr, group=model, color = model_type)) + 
  facet_grid(cols = vars(image_type), rows=vars(metric)) +
  geom_smooth(method = 'gam', se = FALSE) + theme_bw() +
  labs(x = 'Correlation', y = 'Model Layer Depth (Relative)')

#####---------------------------------------------------------------------

response_data <- read_csv('aesthetic_responses.csv') %>%
  group_by(Subj, ImageType, Image) %>%
  summarize(rating = mean(Rating), RT = mean(RT)) %>%
  magrittr::set_colnames(c('subject','image_type','image','rating','reaction_time')) %>%
  group_by(image, image_type) %>% summarise(rating = mean(rating))

activity <- read_csv('alexnet_special.csv') %>%
  left_join(response_data)

activity %>% filter(image_type %in% c('lsc')) %>%
  #filter(str_detect(model_layer, 'Conv2d')) %>%
  ggplot(aes(mean_activity, rating, color = train_type)) + 
  geom_point() + geom_smooth(method = 'lm') +
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free_x') 

activity %>% filter(image_type %in% c('fac')) %>%
  ggplot(aes(mean_activity, rating, color = image_type)) + 
  geom_point() + geom_smooth(method = 'lm') +
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free_x') 

activity %>% filter(str_detect(image_source, 'imagenet')) %>%
  ggplot(aes(mean_activity, sparseness)) + geom_point() + 
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free') + 
  scale_color_gradient(low = 'violet', high = 'cyan') +
  geom_point(aes(color = rating), activity %>% 
               filter(image_type == 'lsc' & image_source == 'vessel'))

activity %>% filter(str_detect(image_source, 'imagenet')) %>%
  ggplot(aes(mean_activity, sparseness)) + geom_point() + 
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free') + 
  scale_color_gradient(low = 'violet', high = 'cyan') +
  geom_point(aes(color = rating), activity %>% 
               filter(image_type == 'fac' & image_source == 'vessel'))

activity %>% filter(str_detect(image_source, 'imagenet')) %>%
  ggplot(aes(mean_activity, sparseness)) + geom_point() + 
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free') + 
  scale_color_gradient(low = 'violet', high = 'cyan') +
  geom_point(aes(color = rating), activity %>% 
               filter(image_type == 'art' & image_source == 'vessel'))

activity %>% filter(image_source != 'vessel') %>%
  ggplot(aes(mean_activity, sparseness, color = image_source)) + 
  geom_point() + facet_wrap(~model_layer_depth, ncol = 6, scales = 'free')

activity %>% replace_na(list(image_type = 'imagenet')) %>%
  filter(image_type %in% c('fac','lsc','art','imagenet')) %>%
  filter(image_source %in% c('vessel', 'imagenet_val')) %>%
  ggplot(aes(sparseness, color = interaction(image_type))) + geom_density() +
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free') 

activity %>% filter(image_type %in% c('lsc')) %>%
  group_by(model, train_type, model_layer_depth) %>%
  cor_test(mean_activity, rating) %>% group_by(train_type) %>%
  summarise(cor_mean = mean(cor), cor_max = max(cor), cor_min = min(cor))

activity %>% replace_na(list(image_type = 'imagenet')) %>%
  filter(image_type %in% c('lsc','imagenet')) %>%
  filter(image_source %in% c('vessel', 'imagenet_val')) %>%
  mutate(rating_bin = cut(rating, breaks = quantile(rating, probs = seq(0, 1, 0.5), na.rm = TRUE))) %>%
  {ggplot(., aes(mean_activity)) + geom_density(fill='gray', alpha = 0.5) +
  facet_wrap(~model_layer_depth, ncol = 6, scales = 'free_x') +
      geom_density(data = . %>% filter(!is.na(rating_bin)), 
                   aes(mean_activity, fill = rating_bin), alpha = 0.5) + labs(fill = 'ratings_bin')}
  
#####---------------------------------------------------------------------

normalize <- function(x) {return((x- min(x)) /(max(x)-min(x)))}

metric_data <- results_raw_wide %>% filter(model == 'alexnet') %>%
  mutate_at(vars(mean_activity:sparseness), normalize)

metric_data %>% pivot_longer(cols = contains('imagenet') & contains('cosine'), 
                             names_to = 'metric', values_to = 'score') %>%
  ggplot(aes(score, rating, color = metric)) + 
  geom_point() + geom_smooth(method='lm', se = FALSE) +
  facet_wrap(~model_layer_index, ncol = 6, scales = 'free_x')

metric_data %>% pivot_longer(cols = contains('imagenet') & contains('cosine'), 
                             names_to = 'metric', values_to = 'score') %>%
  ggplot(aes(score, color = metric)) + geom_density() +
  facet_wrap(~model_layer_index, ncol = 6, scales = 'free_x')

metric_data %>% ggplot(aes(mean_distance_to_imagenet, rating)) +
  geom_point() + geom_smooth(method='lm') +
  facet_wrap(~model_layer_index, ncol = 6, scales = 'free_x') 

#####---------------------------------------------------------------------

rankings <- results_corr_long %>% group_by(model, metric, image_type) %>%
  mutate(corr_abs = abs(corr)) %>% filter(corr_abs == max(corr_abs))

rankings %>% group_by(model, image_type) %>% filter(image_type == 'art') %>%
  filter(corr_abs == max(corr_abs)) %>% group_by(metric) %>% tally()

metric_counts <- rankings %>% group_by(model, image_type) %>% 
  filter(corr_abs == max(corr_abs)) %>% 
  group_by(model, metric) %>% 
  summarise(metric_count = n())

rankings %>% group_by(model, image_type) %>% 
  filter(corr_abs == max(corr_abs)) %>%
  left_join(results_corr_long %>% group_by(model) %>% 
              summarise(model_depth = n_distinct(model_layer_index))) %>%
  mutate(sparsity = ifelse(metric == 'sparseness', 1, 0)) %>%
  select(model, model_depth, sparsity) %>%
  ggplot(aes(model_depth, sparsity)) + geom_jitter(height = 0) +
  geom_smooth(method='glm', method.args = list(family = "binomial"))

results_corr_long %>% group_by(model) %>% 
  summarise(depth = n_distinct(model_layer_index))

rankings %>% group_by(model, image_type) %>% 
       filter(corr_abs == max(corr_abs)) %>% View()

#####---------------------------------------------------------------------

pacman::p_load('voxel')

neural_reg_results <- read_csv('neural_reg_results.csv')

neural_reg_results$image_type <- as.factor(neural_reg_results$image_type)

neural_reg_gam <- gam(score ~ image_type + s(model_layer_depth, by = image_type, bs = 'cs'), 
                      data = neural_reg_results)

summary(neural_reg_gam)

neural_reg_lm <- lm(score ~ image_type*model_layer_depth, data = neural_reg_results)

summary(neural_reg_lm)

plotGAM(neural_reg_gam, smooth.cov = 'model_layer_depth', groupCovs = c('image_type'))

ggplot(neural_reg_results, aes(model_layer_depth, score, color = image_type)) +
  geom_smooth(method='gam', lty = 1) + geom_smooth(method='lm', lty=2) + theme_bw()

neural_reg_results %>% filter(model == 'alexnet' & train_type == 'imagenet') %>%
  ggplot(aes(model_layer_depth, score, color = image_type)) + 
  geom_smooth(method='gam', lty = 1) + geom_smooth(method='lm', lty=2) + theme_bw()


#####---------------------------------------------------------------------

results_corr_long %>% group_by(model, metric, image_type) %>%
  mutate(corr_abs = abs(corr)) %>% filter(corr_abs == max(corr_abs))

results_corr_long %>% mutate(corr_abs = abs(corr)) %>% 
  group_by(model, metric) %>% 
  summarise(corr_abs = mean(corr_abs)) %>%
  filter(corr_abs == max(corr_abs)) %>% View()

results_corr_long %>% mutate(corr_abs = abs(corr)) %>% 
  group_by(model, metric) %>% 
  summarise(corr_abs = max(corr_abs)) %>%
  filter(corr_abs == max(corr_abs)) %>% View()

  

  

  
  
