mmo_corrs %>% mutate(score = corr, model = as.character(subject), train_type = 'human') %>%
  bind_rows(reg_results_max) %>% 
  filter(dataset == 'Oasis', image_type == 'Combo', 
         train_type %in% train_type_labels) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels),
         train_type = factor(train_type, levels = train_type_levels)) %>%
  ggplot(aes(x = image_type, y = score, color = train_type)) + theme_classic() + 
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
              filter(dataset == 'Oasis', image_type == 'Combo')) +
  labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
       x = element_blank(), fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 24), legend.text = element_text(size=20),
        axis.title.y = element_markdown(size = 20), legend.text.align = 0.5) +
  scale_fill_manual(values = magma(n = 4, end = 0.8)) + 
  scale_color_manual(values = magma(n = 4, end = 0.8)) + 
  guides(fill=guide_legend(nrow=1, byrow=TRUE)) +
  theme(strip.background.x = element_blank(), strip.text.x = element_text(color = 'white')) +
  geom_hline(yintercept = 0, lty = 2) + scale_y_continuous(limits = c(-0.25,1.0), expand = c(0,0),
                                                           breaks = seq(-0.25,1.0,0.25))

best_models <- read_csv('best_models.csv') %>% 
  filter(score_type == 'pearson_r') %>%
  mutate(image_type = 'Combo') %>% drop_na()

best_models %>% 
  mutate(train_type = factor(train_type, levels = train_type_levels)) %>%
  ggplot(aes(x = image_type, y = score, fill = train_type)) + 
  facet_wrap(~factor(measurement, measurement_levels)) +
  geom_col(position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                position = position_dodge(width=0.9), width = 0.1, lty = 1) +
  easy_remove_x_axis() + custom_themes$bottom_legend +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), 
            alpha = 0.3, fill = 'black', inherit.aes = FALSE,
            data = reliability_combo %>% 
              filter(reliability == 'Split-Half') %>%
              filter(dataset == 'Oasis', image_type == 'Combo')) +
  labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
       x = element_blank(), fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 24), legend.text = element_text(size=20),
        axis.title.y = element_markdown(size = 20), legend.text.align = 0.5) +
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
  theme(text = element_text(size = 24), legend.text = element_text(size=24),
        axis.title.y = element_text(size = 18), legend.text.align = 0.5) +
  scale_color_manual(values = magma(n = 4, end = 0.8)[2:3]) +
  labs(x = 'Model Layer Depth (Binned)', 
       y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)') +
  guides(fill = 'none', color = 'none') +
  theme(axis.title.y = element_markdown(),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), 
            alpha = 0.3, fill = 'black', inherit.aes = FALSE,
            data = reliability_combo %>% 
              filter(reliability == 'Split-Half') %>%
              filter(dataset == 'Oasis', image_type == 'Combo')) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  theme(strip.background.x = element_rect(fill = '#E6EAED')) +
  geom_hline(yintercept = 0, lty = 2) #+ #scale_y_continuous(limits = c(0.0,1.0), expand = c(0,0),
                                                           #breaks = seq(0.0,1.0,0.25))

mmo_reg_combo %>% group_by(subject, measurement, image_type, dataset) %>%
  summarise(n = n(), score = mean(score, na.rm = TRUE), 
            mmo_corr = mean(corr, na.rm = TRUE)) %>%
  ungroup() %>% filter(dataset == 'Oasis', image_type == 'Combo') %>%
  ggplot(aes(x = score, y = mmo_corr, group = measurement)) + theme_bw() +
  facet_wrap(~measurement) + geom_point() + geom_smooth(method = 'lm') +
  stat_cor(cex = 6.5, label.x = -0.15, label.y = 0.95, show.legend = FALSE) +
  labs(y = '*r<sub>Pearson</sub>*(Individual, Group Ratings)',
       x = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)', 
       fill = element_blank(), color = element_blank()) +
  theme(text = element_text(size = 24), legend.text.align = 0.5,
        axis.title.x = element_markdown(size = 24),
        axis.title.y = element_markdown(size = 24)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  theme(strip.background.x = element_rect(fill = '#E6EAED'))

subject_regs %>% group_by(subject, dataset, measurement, image_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  mutate(statistic = 'Average') %>%
  bind_rows(subject_regs %>% filter(model == 'ViT-L/14') %>%
              filter(dataset == 'Oasis', image_type == 'Combo') %>%
              mutate(statistic = 'Best')) %>%
  ggplot(aes(x = statistic, y = score)) + theme_bw() +
  facet_wrap(~ measurement) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  geom_boxplot(position = position_dodge(width=0.9), 
               outlier.shape = NA, width = 0.5) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar',
               width = 0.5, lty = 1, fill = 'black', alpha = 0.3) + 
  custom_themes$bottom_legend + #easy_remove_legend() +
  labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
       x = 'Model Selection', fill = element_blank(), color = element_blank()) +
  scale_y_continuous(limits = c(-0.25,0.85), breaks = seq(-0.25,0.75,0.25)) +
  theme(text = element_text(size = 24),
        axis.title.y = element_markdown(size = 24), legend.text.align = 0.5) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  theme(strip.background.x = element_rect(fill = '#E6EAED'))
  
