### Analysis Scratchpad -----------------------------------------------------------

#* Ranking Options ---------------------------------------------------------------

shared_model_regex <- 'vit|ViT|resnet50|RN50|resnet101|RN101'

reg_results_max %>% group_by(model, train_type) %>%
  summarise(n = n(), score_ci = mean_cl_boot(score)) %>% 
  ungroup() %>% unnest(score_ci) %>% filter(n == 21) %>%
  #filter(str_detect(model, 'convnext|swin|vit')) %>%
  #filter(str_detect(model, shared_model_regex)) %>%
  #filter(str_detect(model, 'resnet50|RN50')) %>%
  #filter(str_detect(train_type, 'semi-weakly')) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  mutate(rank = dense_rank(-y)) %>% arrange(rank) %>% print(n = 30)

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type) %>%
  summarise(n = n(), score_ci = mean_cl_boot(explained_variance)) %>% 
  ungroup() %>% unnest(score_ci) %>% filter(n == 21) %>%
  mutate_at(vars(y, ymin, ymax), round, 3) %>%
  mutate(report = paste0(y,' [',ymin,', ',ymax,']')) %>%
  mutate(rank = dense_rank(-y)) %>% arrange(rank)

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(str_detect(model, shared_model_regex)) %>%
  select(model, train_type, dataset, measurement, 
         image_type, explained_variance) %>%
  group_by(dataset, measurement, image_type) %>% 
  mutate(rank = dense_rank(-explained_variance)) %>% 
  filter(image_type == 'Combo') %>% arrange(rank)

reg_results_max %>% left_join(splithalf_summary) %>%
  filter(str_detect(model, shared_model_regex)) %>%
  select(model, train_type, dataset, measurement, 
         image_type, score) %>% arrange(desc(score))

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  #filter(str_detect(model, shared_model_regex)) %>%
  select(model, train_type, dataset, measurement, 
         image_type, explained_variance) %>%
  arrange(desc(explained_variance))

### Figure Scratchpad -----------------------------------------------------------

#* Overall Scores --------------------------------------------------

train_type_titles <- c('Human Subjects (Mean-Minus-One)' = 'human')
train_type_levels <- c('Human Subjects (Mean-Minus-One)')
mmo_corrs %>% mutate(model = as.character(subject), train_type = 'human') %>%
  mutate(measurement = str_to_title(measurement)) %>%
  filter(dataset == 'oasis', image_type == 'Combo', 
         train_type %in% train_type_titles) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_titles),
         train_type = factor(train_type, train_type_levels)) %>%
  ggplot(aes(x = factor(measurement, levels = measurement_levels), y = score)) +
  geom_jitter(width = 0.25, alpha = 0.25) + geom_boxplot(outlier.shape = NA, width = 0.5) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar', 
               width = 0.5, lty = 1, alpha = 0.3, fill = 'gray') +
  labs(x = 'Image Type', y = 'Score (Pearson R)', 
       color = 'Subject Type', fill = 'Subject Type') + custom_themes$bottom_legend + 
  theme(text = element_text(size=18)) + labs(x = element_blank(), fill = element_blank(), 
                                             color = element_blank())

expression('r[Pearson]')
TeX('$r_{Pearson}$')

reg_results_max %>% bind_rows(mmo_scores) %>% 
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  mutate(measurement = str_to_title(measurement)) %>%
  filter(dataset == 'oasis', image_type == 'Combo', 
         train_type %in% train_type_titles) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_titles),
         train_type = factor(train_type, levels = train_type_levels),
         image_type = factor(image_type, levels = oasis_levels)) %>%
  ggplot(aes(x = image_type, y = explained_variance, color = train_type)) +
  geom_hline(yintercept = 0.5, lty = 2, alpha = 0.5) +
  geom_point(alpha = 0.3, position = position_jitterdodge(dodge.width=0.9)) +
  facet_wrap(~factor(measurement, levels = measurement_levels)) + #geom_boxplot(outlier.shape = NA) +
  geom_boxplot(position = position_dodge(width=0.9), 
               outlier.shape = NA, width = 0.5) +
  stat_summary(aes(fill = train_type), fun.data = median_cl_boot, geom = 'crossbar',
               position = position_dodge(width=0.9), 
               width = 0.5, lty = 1, alpha = 0.3) +
  labs(x = 'Image Type', y = 'Score (Pearson R)', 
       color = 'Subject Type', fill = 'Subject Type') + custom_themes$bottom_legend + 
  labs(x = element_blank(), fill = element_blank(), color = element_blank()) +
  easy_remove_x_axis() + theme(text = element_text(size = 24),
                               axis.title.y = element_text(size = 18)) +
  labs(y = "% Explainable Variance Explained")

reg_results_max %>% bind_rows(mmo_scores) %>% 
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  mutate(measurement = str_to_title(measurement)) %>%
  mutate(dataset = str_to_title(dataset)) %>%
  filter(image_type != 'Combo') %>%
  group_by(dataset) %>%
  mutate(image_type = as.factor(image_type)) %>% ungroup() %>%
  mutate(image_type_numeric = as.numeric(image_type)) %>% #select(image_type, image_type_numeric) %>% distinct()
  filter(image_type != 'Combo',  train_type %in% c('imagenet','selfsupervised','imagenet21k')) %>%
  ggplot(aes(x = image_type_numeric, y = explained_variance, color = measurement)) +
  geom_point(alpha = 0.3, position = position_jitterdodge(jitter.width = 0.1, dodge.width=0.9)) +
  facet_wrap(~factor(dataset), scales = 'free_x') +
  geom_boxplot(aes(group = interaction(measurement,image_type)), position = position_dodge(width=0.9), 
               outlier.shape = NA, width = 0.5) +
  stat_summary(aes(fill = measurement), fun.data = median_cl_boot, geom = 'crossbar',
               position = position_dodge(width=0.9), 
               width = 0.5, lty = 1, alpha = 0.3) +
  labs(x = 'Image Type', y = 'Score (Pearson R)', 
       color = 'Subject Type', fill = 'Subject Type') + custom_themes$bottom_legend + 
  labs(x = element_blank(), fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 24), axis.title.y = element_text(size = 18)) +
  scale_x_continuous(breaks = my_breaks, labels = my_labels) +
  labs(y = "*% Explainable Variance Explained")

#* Image Type Differences --------------------------------------------------

my_breaks <- function(x) { if (max(x) < 5) c(1,2,3,4) else c(5,6,7,8,9)}
my_labels <- function(x) { if (max(x) < 5) c('Person','Animal','Object','Scene') else 
  c('Art','Buildings','Interiors','Landscapes','Faces')}

reg_results_max %>% bind_rows(mmo_scores) %>%  filter(image_type != 'Combo') %>%
  mutate(image_type_numeric = as.numeric(image_type)) %>% 
  filter(image_type != 'Combo',  train_type %in% c('imagenet','selfsupervised','imagenet21k')) %>%
  ggplot(aes(x = image_type_numeric, y = score, color = measurement)) +
  facet_wrap(~factor(dataset), scales = 'free_x') +
  geom_point(alpha = 0.3, position = position_jitterdodge(jitter.width = 0.3, dodge.width=0.9)) +
  geom_boxplot(aes(group = interaction(measurement,image_type)), 
               position = position_dodge(width=0.9), outlier.shape = NA, width = 0.5) +
  stat_summary(aes(fill = measurement), geom = 'crossbar', fun.data = median_cl_boot, 
               position = position_dodge(width=0.9), width = 0.5, lty = 1, alpha = 0.3) +
  geom_rect(aes(xmin = image_type_numeric-0.3, xmax = image_type_numeric+0.3, 
                ymin = lower, ymax = upper, fill = measurement), alpha = 0.3,
            position = position_dodge(width = 0.9), show.legend = FALSE,
            data = reliability_combo %>% filter(image_type != 'Combo') %>%
              mutate(image_type_numeric = as.numeric(image_type)) %>%
              filter(reliability == 'Split-Half'), inherit.aes = FALSE) +
  scale_x_continuous(breaks = my_breaks, labels = my_labels) + custom_themes$bottom_legend +
  labs(y = "*r<sub>Pearson</sub>*(Predicted, Actual Ratings)",
       x = element_blank(), fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 24), axis.title.y = element_markdown(size = 18)) 

#* Trained versus Untrained --------------------------------------------------

reg_results_max %>% group_by(model, train_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  filter(train_type %in% c('imagenet','random')) %>%
  group_by(model) %>% summarise(count = n()) %>% filter(count < 2)

train_type_labels <- c('Trained Models' = 'imagenet', 
                       'Untrained Models' = 'random')
reg_results_max %>% filter(train_type %in% c('imagenet','random')) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels)) %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  filter(!model %in% c('convnext_base')) %>%
  ggwithinstats(x = train_type, y = score, type = 'parametric', ggtheme = theme_bw(), 
                centrality.label.args = list(size  = 8),
                ggplot.component = list(theme(text = element_text(size = 24),
                                              axis.title.y = element_markdown()),
                                        labs(x = element_blank(),
                                             y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)')))

#* Self- versus Category-Supervised --------------------------------------------------

train_type_labels <- c('Category-Supervised Models' = 'imagenet', 
                       'Self-Supervised Models' = 'selfsupervised')
reg_results_max %>% filter(train_type %in% c('imagenet', 'selfsupervised')) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels)) %>%
  filter(!str_detect(model, noncontrastive_regex)) %>%
  group_by(model, train_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  ggbetweenstats(x = train_type, y = score, type = 'np', ggtheme = theme_bw(),
                 centrality.label.args = list(size  = 8),
                 ggplot.component = list(theme(text = element_text(size = 24),
                                               axis.title.y = element_markdown()),
                                         labs(x = element_blank(),
                                              y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)')))

#* Cross-Decoding --------------------------------------------------

cross_decoding_overall %>% select(model, train_type, train_data, test_data, cross_score) %>%
  mutate(train_data = fct_recode(factor(train_data), !!!reverse_labels(cross_decoding_labels)),
         test_data = fct_recode(factor(test_data), !!!reverse_labels(cross_decoding_labels))) %>%
  filter(train_data != test_data) %>%
  filter(str_detect(train_data, 'Scene|Landscapes|Art|Faces'),
         str_detect(test_data, 'Scene|Landscapes|Art|Faces')) %>%
  group_by(model, train_type, train_data) %>% 
  summarise(cross_score = mean(cross_score)) %>%
  filter(str_detect(train_data, 'Scene|Landscapes|Art|Faces')) %>%
  ggplot(aes(x = train_data, y = cross_score)) + 
  stat_summary(fun.data = median_cl_boot, geom = 'bar', color = 'black',
               width = 0.5, lty = 1, alpha = 0.3) +
  stat_summary(fun.data = median_cl_boot, geom = 'errorbar', color = 'black',
               width = 0.5, lty = 1, alpha = 0.3) +
  theme(text = element_text(size = 24)) +
  labs(y = 'Cross-Decoding Score', x = 'Training Data')

cross_decoding_overall %>% select(model, train_type, train_data, test_data, cross_score) %>%
  mutate(train_data = fct_recode(factor(train_data), !!!reverse_labels(cross_decoding_labels)),
         test_data = fct_recode(factor(test_data), !!!reverse_labels(cross_decoding_labels))) %>%
  filter(train_data != test_data, !str_detect(train_type, 'clip|slip')) %>%
  filter(str_detect(train_data, 'Scene|Landscapes|Art|Faces'),
         str_detect(test_data, 'Scene|Landscapes|Art|Faces')) %>%
  group_by(model, train_type, train_data, test_data) %>% 
  mutate(score = cross_score) %>% summarise(score = mean(score)) %>%
  filter(str_detect(train_data, 'Landscapes|Scene'),
         str_detect(test_data, 'Landscapes|Scene')) %>%
  mutate(train_data = str_replace(train_data, 'V: |O: ', ''),
         test_data = str_replace(test_data, 'V: |O: ', '')) %>%
  mutate(scoring = 'cross_decoding') %>%
  bind_rows(reg_results_max %>% filter(measurement == 'Beauty') %>%
              filter(train_type %in% c('imagenet','selfsupervised')) %>%
              filter(str_detect(image_type, 'Scene|Landscapes')) %>%
              mutate(train_data = image_type, test_data = image_type) %>%
              select(model, train_type, train_data, test_data, score) %>% 
              mutate(scoring = 'loocv')) %>%
  ggplot(aes(x = train_data, y = score)) + facet_wrap(~scoring) + 
  stat_summary(fun.data = median_cl_boot, geom = 'bar', color = 'black',
               width = 0.5, lty = 1, alpha = 0.3) +
  stat_summary(fun.data = median_cl_boot, geom = 'errorbar', color = 'black',
               width = 0.5, lty = 1, alpha = 0.3) +
  theme(text = element_text(size = 24)) +
  labs(y = 'Cross-Decoding Score', x = 'Training Data')

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(model %in% model_subset) %>%
  mutate(model = factor(model, levels=model_subset)) %>%
  mutate(model = fct_recode(model, !!!model_subset)) %>%
  ggplot(aes(x = model, y = explained_variance)) + 
  stat_summary(fun.data = median_cl_boot, geom = 'bar', color = 'black',
               width = 1.0, lty = 1, alpha = 0.3) +
  stat_summary(fun.data = median_cl_boot, geom = 'errorbar', color = 'black',
               width = 1.0, lty = 1, alpha = 0.3) +
  theme(text = element_text(size = 24)) +
  geom_text(aes(y = 0.05, x = model, label = model), size = 7, hjust = 0, check_overlap = TRUE) + 
  labs(y = '% Explainable Variance Explained', x = '') + 
  scale_x_discrete(labels = rev(model_labels), limits = rev) + coord_flip() +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0,1.0,0.2), limits = c(0,1.0))


#* SLIP Results --------------------------------------------------

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(train_type == 'slip', !str_detect(model, 'Ep100|CC12M')) %>%
  mutate(slip_kind = str_split(model, '-', simplify = TRUE)[, 3]) %>%
  mutate(slip_size = str_split(model, '-', simplify = TRUE)[, 2],
         slip_size = fct_recode(factor(slip_size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  mutate(model = paste0('ViT-',slip_size)) %>%
  mutate(model = factor(model, levels = c('ViT-Small','ViT-Base','ViT-Large'))) %>%
  select(model, train_type, measurement, image_type, dataset, 
         explained_variance, slip_kind, slip_size) %>%
  ggplot(aes(x = model, y = explained_variance, fill = slip_kind)) + 
  stat_summary(fun.data = median_cl_boot, geom = 'bar', color = 'black',
               position = position_dodge(width = 0.9),
               width = 0.9, lty = 1, alpha = 0.3) +
  stat_summary(fun.data = median_cl_boot, geom = 'errorbar', color = 'black',
               position = position_dodge(width = 0.9),
               width = 0.9, lty = 1, alpha = 0.3) +
  theme(text = element_text(size = 24)) + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = '', fill = '') + coord_flip()

#* Language Gains --------------------------------------------------

reg_results_max %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  mutate(language = ifelse(str_detect(train_type, 'clip|slip'), 'Language','No Language')) %>%
  select(model, train_type, dataset, measurement, 
         image_type, language, explained_variance) %>%
  group_by(dataset, measurement, image_type, language) %>%
  mutate(rank = dense_rank(-explained_variance)) %>% filter(rank == 1) %>%
  arrange(image_type, -explained_variance, language) %>% #print(n = 100)
  select(-model, -train_type, -rank) %>%
  pivot_wider(names_from = language, values_from = explained_variance) %>%
  mutate(difference = Language - `No Language`) %>%
  mutate(dataset_measurement = paste(dataset, measurement, sep = ' - ')) %>%
  ggplot(aes(x = image_type, y = difference)) + 
  geom_col(aes(fill = measurement), position = position_dodge(width = 0.9)) +
  facet_wrap(~dataset, scales = 'free_x', ncol = 4) +
  labs(y = 'Gain in % Explainable \n Variance Explained', x = '', fill = '') +
  custom_themes$bottom_legend +
  theme(text = element_text(size = 24))

boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  mutate(language = ifelse(str_detect(train_type, 'clip'), 'Language','No Language')) %>%
  select(model, train_type, bootstrap_ids, dataset, measurement, 
         image_type, language, explained_variance) %>%
  group_by(bootstrap_ids, dataset, measurement, image_type, language) %>%
  mutate(rank = dense_rank(-explained_variance)) %>% filter(rank == 1) %>%
  group_by(bootstrap_ids, dataset, measurement, image_type, language) %>%
  select(-model, -train_type, -rank) %>% ungroup() %>%
  pivot_wider(names_from = language, values_from = explained_variance, values_fn = mean) %>%
  mutate(difference = Language - `No Language`) %>%
  group_by(dataset, measurement, image_type) %>%
  summarise(lower_ci = quantile(difference, 0.025, na.rm = TRUE),
            upper_ci = quantile(difference, 0.975, na.rm = TRUE),
            difference = mean(difference, na.rm = TRUE)) %>% ungroup() %>%
  ggplot(aes(x = image_type, y = difference, group = measurement)) + 
  geom_col(aes(fill = measurement), position = position_dodge(width = 0.9)) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.3, 
                position = position_dodge(width = 0.9)) +
  facet_wrap(~dataset, scales = 'free_x', ncol = 4) +
  labs(y = 'Gain in % Explainable \n Variance Explained', x = '', fill = '') +
  theme(text = element_text(size = 24)) + custom_themes$bottom_legend +
  scale_fill_manual(values = turbo(n = 3, direction = -1, begin = 0.25, end = 0.75))
