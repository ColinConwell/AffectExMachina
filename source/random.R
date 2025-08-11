pacman::p_load('viridis')
  
model_list <- 'RegNet-64Gf|ViT-L/14|ViT-L-|ViT-B-|ViT-S-'

boot_results %>% filter(str_detect(model, model_list)) %>%
  filter(!str_detect(model, 'Ep100|CC12M')) %>%
  select(model, train_type) %>% distinct()

temp_results <- boot_results %>% 
  #filter(str_detect(model, model_list)) %>%
  #filter(!str_detect(model, 'Ep100|CC12M|INFT')) %>%
  filter(dataset != 'Vessel') %>% 
  filter(image_type == 'Combo') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  group_by(model, train_type, bootstrap_id, measurement_plus) %>%
  summarise(score = mean(score)) %>%
  group_by(model, train_type, measurement_plus) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup()


temp_results %>% 
  mutate(kind = str_split(model, '-', simplify = TRUE)[,3]) %>%
  mutate(kind = ifelse(kind == '', 'OpenAI-CLIP', kind)) %>% 
  mutate(size = str_split(model, '-', simplify = TRUE)[,2]) %>% 
  mutate(size = ifelse(size == 'L/14', 'Large-Custom', size)) %>% 
  mutate(size = fct_recode(factor(size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  mutate(kind = factor(kind, levels = c('OpenAI-CLIP', 'SLIP','CLIP','SimCLR'))) %>%
  mutate(model = paste0('ViT-',size)) %>%
  mutate(kind = ifelse(size == '64Gf', 'SEER', as.character(kind)),
         model = ifelse(kind == 'SEER', 'RegNet-64Gf-SEER', model)) %>%
  mutate(model = factor(model, levels = c('ViT-Small','ViT-Base','ViT-Large', 
                                          'ViT-Large-Custom', 'RegNet-64Gf-SEER'))) %>%
  ggplot(aes(x = model, y = explained_variance, fill = kind)) + 
  facet_wrap(~measurement_plus, nrow = 1) +
  geom_col(color = 'black', position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
           width = 0.9, lty = 1, alpha = 0.4) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
                position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
                width = 0.9, lty = 1, alpha = 0.3) +
  geom_text(aes(y = 0.05, x = model, label = model), 
            size = 5, hjust = 0, check_overlap = TRUE) + 
  theme(text = element_text(size = 24), panel.spacing.x = unit(0.5, "lines")) + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = '', fill = '') + coord_flip() +
  scale_fill_manual(values = c('darkgrey', 'blue', plasma(n = 3, direction = -1, begin = 0.2, end = 0.8))) +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0))

temp_results <- boot_results %>% 
  filter(str_detect(model, model_list)) %>%
  filter(!str_detect(model, 'Ep100|CC12M|INFT')) %>%
  filter(dataset != 'Vessel') %>% 
  filter(image_type == 'Combo') %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  group_by(model, train_type, bootstrap_id, measurement_plus) %>%
  summarise(score = mean(score)) %>%
  group_by(model, train_type, measurement_plus) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            score = mean(score, na.rm = TRUE)) %>% ungroup()

temp_results %>% 
  mutate(kind = str_split(model, '-', simplify = TRUE)[,3]) %>%
  mutate(kind = ifelse(kind == '', 'OpenAI-CLIP', kind)) %>% 
  mutate(size = str_split(model, '-', simplify = TRUE)[,2]) %>% 
  #mutate(size = ifelse(size == 'L/14', 'Large-Custom', size)) %>% 
  mutate(size = fct_recode(factor(size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  mutate(kind = factor(kind, levels = c('OpenAI-CLIP', 'SLIP','CLIP','SimCLR'))) %>%
  mutate(model = paste0('ViT-',size)) %>%
  mutate(kind = ifelse(size == '64Gf', 'SEER', as.character(kind)),
         model = ifelse(kind == 'SEER', 'RegNet-64Gf', model)) %>%
  mutate(model = factor(model, levels = c('ViT-Small','ViT-Base','ViT-Large', 
                                          'ViT-L/14', 'RegNet-64Gf'))) %>%
  mutate(measurement_plus = str_replace(measurement_plus, 'Oasis - ', '')) %>%
  ggplot(aes(x = model, y = score, fill = kind)) + 
  facet_wrap(~measurement_plus, nrow = 1) +
  geom_col(color = 'black', position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
           width = 0.9, lty = 1, alpha = 0.4) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), color = 'gray', alpha = 0.5,
            data = splithalf_summary %>% filter(dataset == 'Oasis', image_type == 'Combo') %>%
              mutate(measurement_plus = paste(measurement, sep = ' - ')), inherit.aes = FALSE) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
                position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
                width = 0.9, lty = 1, alpha = 0.3) +
  geom_text(aes(y = 0.05, x = model, label = model), 
            size = 5, hjust = 0, check_overlap = TRUE) + 
  theme(text = element_text(size = 24), panel.spacing.x = unit(0.5, "lines")) + custom_themes$bottom_legend +
  labs(y = "*r<sub>Pearson</sub>*(Predicted, Actual Ratings)", x = '', fill = '') + coord_flip() +
  scale_fill_manual(values = c('darkgrey', 'blue', plasma(n = 3, direction = -1, begin = 0.2, end = 0.8))) +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0)) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.title.x = element_markdown())

model_contrast <- c('ViT-L/14', 'RegNet-64Gf-SEER')
boot_results %>% filter(model %in% model_contrast) %>%
  filter(image_type == 'Combo', dataset == 'Oasis',
         score_type == 'pearson_r') %>%
  select(model, measurement, score, bootstrap_id) %>%
  pivot_wider(names_from = model, values_from = score) %>%
  mutate(difference = !!sym(model_contrast[[1]]) - !!sym(model_contrast[[2]])) %>%
  summarise(lower_ci = quantile(difference, 0.025, na.rm = TRUE),
            upper_ci = quantile(difference, 0.975, na.rm = TRUE),
            score = mean(difference, na.rm = TRUE)) %>% ungroup() %>% select(score, lower_ci, upper_ci)
  


boot_results %>% filter(!str_detect(model, 'Ep100|CC12M|INFT')) %>%
  filter(str_detect(train_type, 'seer|slip|clip|selfsupervised')) %>%
  select(model, train_type) %>% distinct() %>% print(n = 40)
