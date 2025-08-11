# to be used with aesthetics/arxiv_results.R

pacman::p_load('ggtext','viridisLite')

train_type_labels <- c('Category-Supervised Models' = 'imagenet', 
                       'Self-Supervised Models' = 'selfsupervised',
                       'Untrained Models' = 'random')

train_type_levels <- c('Category-Supervised Models', 
                       'Self-Supervised Models', 
                       'Untrained Models')

theme_set(theme_bw())

reg_results_max %>%
  filter(dataset == 'Oasis', image_type == 'Combo', 
         measurement == 'Beauty') %>%
  filter(train_type %in% c('seer','selfsupervised')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  select(model, train_type, image_type, measurement, 
         score, explained_variance) %>%
  #arrange(-score) %>% print(n = 30) %>%
  filter(!str_detect(model, 'RotNet|JigSaw|INFT')) %>%
  arrange(-score) %>% print(n = 30) %>%
  summarise(explained_variance = mean_cl_boot(explained_variance))

reg_results_max %>%
  filter(dataset == 'Oasis', image_type == 'Combo', 
         measurement == 'Beauty') %>%
  filter(train_type %in% c('clip','slip')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  select(model, train_type, image_type, measurement, 
         score, explained_variance) %>%
  arrange(-score) %>% print(n = 30)

reg_results_max %>%
  filter(dataset == 'Oasis', image_type == 'Combo', 
         #measurement == 'Beauty',
         train_type %in% train_type_labels) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels),
         train_type = factor(train_type, levels = names(train_type_labels))) %>%
  ggplot(aes(x = image_type, y = score, color = train_type)) + 
  facet_wrap(~factor(measurement, measurement_levels)) +
  geom_point(alpha = 0.3, position = position_jitterdodge(dodge.width=0.9)) +
  geom_boxplot(position = position_dodge(width=0.9), 
               outlier.shape = NA, width = 0.5) +
  stat_summary(aes(fill = train_type), fun.data = median_cl_boot, geom = 'crossbar',
               position = position_dodge(width=0.9), 
               width = 0.5, lty = 1, alpha = 0.3) +
  easy_remove_x_axis() + custom_themes$bottom_legend +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), 
            alpha = 0.3, fill = 'black', inherit.aes = FALSE,
            data = reliability_combo %>% 
              filter(reliability == 'Split-Half') %>%
              #filter(measurement == 'Beauty') %>%
              filter(dataset == 'Oasis', image_type == 'Combo')) +
  labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
       x = element_blank(), fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 30), legend.text = element_text(size=24),
        axis.title.y = element_markdown(size = 30), legend.text.align = 0.5) +
  scale_fill_manual(values = magma(n = 4, end = 0.8)[2:4]) + 
  scale_color_manual(values = magma(n = 4, end = 0.8)[2:4]) + 
  guides(fill=guide_legend(nrow=1, byrow=TRUE)) +
  geom_hline(yintercept = 0, lty = 2) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

boot_results %>% filter(image_type == 'Combo') %>%
  filter(dataset == 'Oasis') %>%
  filter(train_type %in% c('imagenet21k','seer','random')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, measurement, dataset) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  group_by(train_type, measurement, dataset) %>%
  filter(score == max(score))

best_models <- read_csv('top_models.csv') %>%
  mutate(image_type = 'Combo')

train_type_labels <- c('Category-Supervised Models' = 'imagenet21k', 
                       'Self-Supervised Models' = 'seer',
                       'Untrained Models' = 'random')

best_models %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels),
         train_type = factor(train_type, levels = names(train_type_labels))) %>%
  ggplot(aes(x = image_type, y = score, fill = train_type)) + 
  facet_wrap(~factor(measurement, measurement_levels)) +
  geom_col(position = position_dodge(width=1.2), alpha = 0.75) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), 
                position = position_dodge(width=1.2), width = 0.1, lty = 1) +
  easy_remove_x_axis() + custom_themes$bottom_legend +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), 
            alpha = 0.3, fill = 'black', inherit.aes = FALSE,
            data = reliability_combo %>% 
              filter(reliability == 'Split-Half') %>%
              filter(dataset == 'Oasis', image_type == 'Combo')) +
  labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
       x = element_blank(), fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 30), legend.text = element_text(size=24),
        axis.title.y = element_markdown(size = 30), legend.text.align = 0.5) +
  scale_fill_manual(values = magma(n = 4, end = 0.8)[2:4]) + 
  scale_color_manual(values = magma(n = 4, end = 0.8)[2:4]) + 
  guides(fill=guide_legend(nrow=1, byrow=TRUE)) +
  geom_hline(yintercept = 0, lty = 2) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
  
reg_results %>% filter(dataset == 'Oasis', image_type == 'Combo') %>% 
  filter(train_type == 'imagenet' | train_type == 'selfsupervised') %>%
  #filter(str_detect(train_type, 'imagenet|selfsupervised')) %>%
  mutate(depth_binned = cut(model_layer_depth, breaks=seq(0,1.0,0.1), labels=seq(0.1,1.0,0.1))) %>%
  group_by(model, train_type, measurement, image_type, depth_binned) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  ggplot(aes(x = depth_binned, y = score)) + theme_bw() +
  facet_wrap(~factor(measurement, measurement_levels)) +
  geom_jitter(aes(color = train_type), alpha = 0.15, width = 0.25) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar',
               width = 0.5, lty = 1, alpha = 0.3) +
  theme(text = element_text(size = 30), legend.text = element_text(size=24),
        axis.title.y = element_text(size = 30), legend.text.align = 0.5) +
  scale_color_manual(values = magma(n = 4, end = 0.8)[2:3]) +
  labs(x = 'Model Layer Depth (Binned)', 
       y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)') +
  guides(fill = 'none', color = 'none') +
  theme(axis.title.y = element_markdown(),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  #theme(strip.background.x = element_rect(fill = '#E6EAED')) +
  geom_hline(yintercept = 0, lty = 2) + theme(axis.text.x = element_text(size = 16))

reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo', dataset == 'Oasis', 
         train_type %in% c('imagenet','selfsupervised')) %>%
  ggplot(aes(x = measurement, y = explained_variance)) + 
  geom_jitter(aes(color = train_type), size = 3, alpha = 0.15, width = 0.25) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar',
               width = 0.5, lty = 1, alpha = 0.5, fill = 'gray') +
  theme(text = element_text(size = 30), legend.text = element_text(size=24),
        axis.title.y = element_text(size = 30), legend.text.align = 0.5) +
  scale_color_manual(values = magma(n = 4, end = 0.8)[2:3]) +
  labs(x = element_blank(), 
       y = '% Explainable Variance Explained') +
  guides(fill = 'none', color = 'none') +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  #theme(strip.background.x = element_rect(fill = '#E6EAED')) +
  #geom_hline(yintercept = 0, lty = 2) + ylim(c(0,1)) + 
  scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0),
                     labels = function(x) {format(x * 100, digits = 2)})

train_type_labels <- c('Top Self-Supervised Model' = 'seer',
                       'Top Language-Aligned Model' = 'clip')

boot_results %>% filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty') %>%
  filter(dataset == 'Oasis') %>%
  mutate(image_type = 'Combo') %>%
  filter(train_type %in% c('seer','clip')) %>%
  filter(str_detect(model, 'SEER|ViT-L')) %>%
  mutate(train_type = ifelse(train_type == 'clip', paste0('CLIP-',model), train_type)) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, image_type, measurement, dataset) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            score = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  group_by(train_type, image_type, measurement, dataset) %>%
  filter(score == max(score)) %>%
  mutate(train_type = ifelse(str_detect(train_type, 'CLIP'), 'clip', train_type)) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels),
         train_type = factor(train_type, levels = names(train_type_labels))) %>%
  ggplot(aes(x = reorder(interaction(model, train_type), score), y = score, fill = train_type)) + 
  facet_wrap(~factor(measurement, measurement_levels)) +
  geom_col(position = position_dodge(width=1.2), alpha = 0.75) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), 
                position = position_dodge(width=1.2), width = 0.1, lty = 1) +
  easy_remove_x_axis() + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained',
       fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 30), legend.text = element_text(size=24),
        axis.title.y = element_text(size = 30), legend.text.align = 0.5) +
  scale_fill_manual(values = c(magma(n = 4, end = 0.8)[3], magma(n = 4, end = 0.8)[1])) + 
  scale_color_manual(values = c(magma(n = 4, end = 0.8)[3], magma(n = 4, end = 0.8)[1])) + 
  guides(fill=guide_legend(nrow=1, byrow=TRUE)) +
  geom_hline(yintercept = 0, lty = 2) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

boot_results %>% filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty') %>%
  filter(dataset == 'Oasis') %>%
  mutate(image_type = 'Combo') %>%
  filter(train_type %in% c('seer','clip')) %>%
  #filter(str_detect(model, 'SEER|ViT-L')) %>%
  mutate(model = ifelse(train_type == 'clip', paste0('CLIP-', model), model)) %>%
  mutate(train_type = ifelse(train_type == 'clip', paste0('CLIP-',model), train_type)) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, image_type, measurement, dataset) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            score = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  group_by(train_type, image_type, measurement, dataset) %>%
  filter(score == max(score)) %>%
  mutate(train_type = ifelse(str_detect(train_type, 'CLIP'), 'clip', train_type)) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels),
         train_type = factor(train_type, levels = names(train_type_labels))) %>%
  ggplot(aes(x = reorder(model, score), y = score, fill = train_type)) + 
  facet_wrap(~factor(measurement, measurement_levels)) +
  geom_col(position = position_dodge(width=1.2), 
           alpha = 0.4, color = 'black') +
  geom_text(aes(y = 0.05, x = model, label = model), 
            size = 8, hjust = 0, check_overlap = TRUE) + 
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), 
                position = position_dodge(width=1.2), 
                width = 0.1, lty = 1, alpha = 1.0) +
  easy_remove_y_axis() + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = element_blank(),
       fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 30), legend.text = element_text(size=24),
        axis.title.y = element_text(size = 30), legend.text.align = 0.5) +
  scale_fill_manual(values = c(magma(n = 4, end = 0.8)[3], 'darkgrey')) + 
  scale_color_manual(values = c(magma(n = 4, end = 0.8)[3], 'darkgrey')) + 
  guides(fill=guide_legend(nrow=1, byrow=TRUE)) + coord_flip() + 
  scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0),
                     labels = function(x) {format(x * 100, digits = 2)}) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  group_by(model, train_type, bootstrap_id, measurement) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type, measurement) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  process_slip_results() %>% 
  ggplot(aes(x = model, y = explained_variance, fill = slip_kind)) + 
  facet_wrap(~measurement, nrow = 1) +
  geom_col(color = 'black', position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
           width = 0.9, lty = 1, alpha = 0.4) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
                position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
                width = 0.9, lty = 1, alpha = 0.3) +
  geom_text(aes(y = 0.05, x = model, label = model), 
            size = 5, hjust = 0, check_overlap = TRUE) + 
  theme(text = element_text(size = 30), panel.spacing.x = unit(0.5, "lines")) + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = '', fill = '') + 
  scale_fill_manual(values = c('darkgrey', mako(n = 3, direction = -1, begin = 0.2, end = 0.8))) +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0),
                                            labels = function(x) {format(x * 100, digits = 2)}) +
  coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())



process_slip_results <- function(results) {
  results %>% filter(model == 'ViT-L/14' | str_detect(train_type, 'slip'), 
                     !str_detect(model, 'Ep100|CC12M')) %>%
    mutate(slip_kind = str_split(model, '-', simplify = TRUE)[, 3],
           slip_kind = ifelse(slip_kind == '', 'OpenAI-CLIP', slip_kind),
           slip_size = str_split(model, '-', simplify = TRUE)[, 2],
           slip_size = ifelse(slip_size == 'L/14', 'Large-Custom', slip_size),
           slip_size = fct_recode(factor(slip_size), 'Base'='B', 'Small'='S', 'Large'='L'),
           slip_kind = factor(slip_kind, 
                              levels = c('OpenAI-CLIP', 'SimCLR', 'CLIP', 'SLIP')),
           model = factor(paste0('ViT-',slip_size), 
                          levels = c('ViT-Small','ViT-Base','ViT-Large', 'ViT-Large-Custom')))
}


reg_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(!str_detect(model_layer, 'Dropout|Attention')) %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  process_slip_results() %>%
  ggplot(aes(x = model_layer_depth, y = explained_variance, color = slip_kind)) + 
  #geom_smooth(aes(group = slip_kind), method = 'gam') + 
  geom_smooth(aes(group = interaction(slip_kind, slip_size)), size = 2, method = 'gam') +
  geom_line(aes(group = interaction(slip_kind, slip_size)), size = 1.5, alpha = 0.15) +
  theme(text = element_text(size = 30), panel.spacing.x = unit(0.5, "lines")) + 
  custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = 'Model Layer Depth', color = '') + 
  scale_color_manual(values = c('darkgrey', plasma(n = 3, direction = -1, begin = 0.2, end = 0.8))) +
  scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0),
                     labels = function(x) {format(x * 100, digits = 2)})

boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  group_by(model, train_type, bootstrap_id, measurement) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type, measurement) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  process_slip_results() %>%
  ggplot(aes(x = model, y = explained_variance, fill = slip_kind)) + 
  facet_wrap(~measurement, nrow = 1) +
  geom_col(color = 'black', position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
           width = 0.9, lty = 1, alpha = 0.4) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
                position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
                width = 0.9, lty = 1, alpha = 0.3) +
  theme(text = element_text(size = 30), panel.spacing.x = unit(0.5, "lines")) + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = 'Model Size', fill = '') + 
  scale_fill_manual(values = c('black', plasma(n = 3, direction = 1, begin = 0.2, end = 0.8))) #+
  scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0),
                     labels = function(x) {format(x * 100, digits = 2)})

boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  process_slip_results() %>%
  select(slip_kind, slip_size, dataset, image_type, measurement, 
         bootstrap_id, explained_variance) %>%
  filter(!str_detect(slip_kind, 'OpenAI-CLIP')) %>%
  pivot_wider(names_from = slip_kind, values_from = explained_variance) %>%
  mutate(difference = SLIP - SimCLR) %>%
  group_by(image_type, measurement) %>%
  summarise(count = sum(SLIP > SimCLR), total = n(),
            lower_ci = quantile(difference, 0.025, na.rm = TRUE),
            upper_ci = quantile(difference, 0.975, na.rm = TRUE),
            'difference' = mean(difference, na.rm = TRUE))
  

boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  group_by(model, train_type, bootstrap_id, measurement) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type, measurement) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  process_slip_results() %>% 
  ggplot(aes(x = model, y = explained_variance, fill = slip_kind)) + 
  facet_wrap(~measurement, nrow = 1) +
  geom_col(color = 'black', position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
           width = 0.9, lty = 1, alpha = 0.4) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
                position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
                width = 0.9, lty = 1, alpha = 0.3) +
  geom_text(aes(y = 0.05, x = model, label = model), 
            size = 5, hjust = 0, check_overlap = TRUE) + 
  theme(text = element_text(size = 30), panel.spacing.x = unit(0.5, "lines")) + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = '', fill = '') + 
  scale_fill_manual(values = c('darkgrey', plasma(n = 3, direction = -1, begin = 0.2, end = 0.8))) +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0)) + coord_flip()


boot_results %>% filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  filter(train_type %in% c('clip','seer','imagenet21k')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(score = mean(explained_variance, na.rm = TRUE)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = dense_rank(-score)) %>% arrange(rank) %>%
  mutate_at(vars(score, lower_ci, upper_ci), round, 3) %>%
  mutate(report = paste0(score,' [',lower_ci,', ',upper_ci,']')) %>%
  select(model, train_type, report) %>% print(n = 200)

splithalf_summary %>% filter(measurement == 'Beauty', image_type == 'Combo')

(0.715**2) / (.989**2) 
(0.904**2) / (.989**2) 

cap_core <- dir('../beauty_lang/fresh_results', full.names = TRUE) %>%
  map(read_csv) %>% bind_rows() %>% 
  mutate(dataset = 'Oasis') %>%
  filter(score_type == 'pearson_r') %>%
  mutate(measurement = str_to_title(measurement)) %>%
  fix_factor_levels() %>%
  mutate(model = str_replace(model, 'GITCaption', 'GIT'))

cap_core %>% filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            score = mean(explained_variance, na.rm = TRUE))

cap_core_levels <- c('CLIP-ViT-B/32-over-Images', 
                     'N-Grams-over-Captions',
                     'GPT2-over-Captions',
                     'SBERT-over-Captions',
                     'CLIP+GPT2-over-Captions',
                     'CLIP+SBERT-over-Captions')

cap_core_levels <- c('CLIP-ViT-B/32-over-Images', 
                     'GPT2-over-CLIPCap',
                     'GPT2XL-over-CLIPCap',
                     'SBERT-over-CLIPCap',
                     'GPT2-over-CLIPReward',
                     'GPT2XL-over-CLIPReward',
                     'SBERT-over-CLIPReward',
                     'GPT2-over-GIT',
                     'GPT2XL-over-GIT',
                     'SBERT-over-GIT')


cap_core %>% filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, bootstrap_ids) %>%
  summarise(score = mean(explained_variance, na.rm = TRUE)) %>%
  group_by(model) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup() %T>%
  print() %>% filter(model %in% cap_core_levels) %>%
  mutate_at(vars(model), factor, levels = rev(cap_core_levels)) %>%
  ggplot(aes(score, model)) + theme_bw() + 
  geom_col(color = 'black', fill = 'gray') +
  geom_errorbar(aes(xmin = lower_ci, xmax = upper_ci),
                width = 0.25) +
  labs(x = '% Explainable Variance Explained', y = '') +
  theme(text = element_text(size = 30)) + #xlim(c(0,1)) +
  scale_x_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0),
                     labels = function(x) {format(x * 100, digits = 2)})

cap_layers <- dir('../clip_caption/layer_results', full.names = TRUE) %>%
  map(read_csv) %>% bind_rows() %>% 
  mutate(dataset = 'Oasis') %>%
  filter(score_type == 'pearson_r') %>%
  mutate(measurement = str_to_title(measurement)) %>%
  fix_factor_levels()

degenerate_layers <- c('Embedding-1', 'Embedding-2', 'Linear-1')
cap_layer_levels <- c('CLIP-ViT-B/32-Visual-Encoder',
                      'CLIP-GPT2-Captioner-Caption-Only',
                      'CLIP-GPT2-Captioner-Prefix-Only',
                      'CLIP-GPT2-Captioner-Prefix+Caption')

cap_layers %>% filter(image_type == 'Combo') %>%
  filter(measurement == 'Beauty', dataset == 'Oasis') %>%
  filter(!model_layer %in% degenerate_layers) %>%
  filter(!str_detect(model_layer, 'Dropout')) %>%
  mutate(model_layer_index = model_layer_index) %>%
  group_by(model) %>%
  mutate(model_depth = max(model_layer_index)) %>%
  ungroup() %>%
  mutate(model_layer_depth = model_layer_index / model_depth) %>%
  ungroup %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  mutate(model_type = 'CLIP-Captioner') %>%
  bind_rows(reg_results %>% filter(model == 'ViT-B/32', train_type == 'clip') %>%
              mutate(model = 'CLIP-ViT-B/32-Visual-Encoder') %>%
              filter(image_type == 'Combo') %>%
              filter(measurement == 'Beauty', dataset == 'Oasis') %>%
              filter(!model_layer %in% degenerate_layers) %>%
              filter(!str_detect(model_layer, 'Dropout')) %>%
              group_by(model, train_type) %>%
              mutate(model_depth = max(model_layer_index)) %>%
              ungroup() %>%
              mutate(model_layer_depth = model_layer_index / model_depth) %>%
              ungroup %>% left_join(splithalf_summary) %>%
              mutate(explained_variance = (score**2 / r**2)) %>%
              mutate(model_type = 'CLIP-Visual-Encoder')) %>%
  mutate(model_type = factor(model_type, levels = c('CLIP-Visual-Encoder', 'CLIP-Captioner'))) %>%
  mutate(model = factor(model, levels = cap_layer_levels)) %>%
  ggplot(aes(x = model_layer_depth, y = explained_variance,
             color = model)) + geom_line(size = 1.5) + ylim(c(0,1)) +
  facet_wrap(~model_type) + theme_bw() +
  labs(x = 'Model Layer Depth', color = '',
       y = '% Explainable Variance Explained') +
  custom_themes$bottom_legend +
  guides(color = guide_legend(nrow=3, byrow=TRUE)) +
  theme(text = element_text(size = 30)) + xlim(c(0,1)) +
  scale_y_continuous(limits = c(0,1.0), labels = function(x) {format(x * 100, digits = 3)})

# reg_results %>% filter(model == 'RN50', train_type == 'clip') %>%
#   filter(image_type == 'Combo', dataset == 'Oasis') %>% 
#   write.csv('../beauty_lang/resnet50_clip_results.csv', row.names = FALSE)

  
  
