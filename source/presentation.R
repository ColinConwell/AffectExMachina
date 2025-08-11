graph <- reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo', dataset == 'Oasis', 
         train_type == 'imagenet') %>%
  ggbetweenstats(x = measurement, y = explained_variance,
                 results.subtitle = FALSE, type = 'np',
                 centrality.plotting = FALSE) + 
  theme_bw() + theme(text = element_text(size=24)) + 
  easy_remove_legend() + 
  labs(x = element_blank(), y = 'Explainable Variance Explained')
#graph$layers[[6]]$aes_params$size = 5
graph$layers[[5]]$aes_params$textsize = 7
graph <- graph + stat_summary(aes(label = round(..y.., 3)), fun = median, geom = 'label',
                              position = position_dodge(width = 1), alpha = 1, size = 7) + 
  scale_color_manual(values = rep('#E18742',3)) + labs(caption = element_blank())

graph

graph_data <- reg_results_max %>% left_join(splithalf_summary) %>%
  #mutate(measurement = paste0(measurement, ' (n=72)')) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  bind_rows(mmo_corrs) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo', dataset == 'Vessel',
         train_type %in% c('imagenet', 'random', 'human'))

graph_data %>% distinct(measurement, upper, lower)

pacman::p_load('wesanderson')

graph_data %>% ungroup() %>%
  mutate(train_type = fct_recode(train_type, 'Human Leave-One-Out' = 'human',
                                 'ImageNet-Trained Models' = 'imagenet', 
                                 'Untrained Models' = 'random')) %>%
  ggplot(aes(measurement, score, fill = train_type, color = train_type)) + theme_bw() +
  geom_point(position = position_jitterdodge(dodge.width = 1, jitter.width = 0.3), 
             alpha = 0.3, size = 3) + 
  geom_violin(position = position_dodge(width = 1), color = 'black') +
  geom_boxplot(width = 0.5, position = position_dodge(width = 1), 
               outlier.shape = NA, color = 'black') + 
  geom_tile(data = graph_data %>% distinct(measurement, r, upper, lower), 
            aes(y = r, height = upper-lower, width=0.9),
            color = 'black', fill = alpha('black', 0.25), show.legend = FALSE) +
  stat_summary(aes(label = round(..y.., 3)), fun = median, geom = 'label', size = 7,
               position = position_dodge(width = 1), alpha = 1, show.legend = FALSE, color = 'black') +
  #stat_summary(aes(label = round(..y.., 3)), fun = median, geom = 'label_repel', 
  #             nudge_x = 0.0, min.segment.length = 0.0, direction = c('x'),
  #             position = position_dodge(width = 1), show.legend = FALSE) +
  scale_fill_manual(values = c(rep(alpha('white', 0), 6))) +
  scale_color_brewer(palette = 'Dark2') +
  #scale_color_manual(values = wes_palette(n=3, name="Darjeeling1")) +
  labs(color = element_blank(), x = element_blank(), y = 'Score (Pearson r)') + 
  custom_themes[['bottom_legend']] + theme(text = element_text(size=24)) + guides('fill' = 'none')

graph <- reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(!image_type %in% c('Combo'), dataset == 'oasis', 
         train_type %in% c('imagenet'), measurement == 'Beauty') %>%
  ggbetweenstats(x = image_type, y = explained_variance, type = 'np',
                 results.subtitle = FALSE, centrality.plotting = FALSE,
                 pairwise.comparisons = FALSE)
graph$layers[[5]]$aes_params$textsize = 5
graph <- graph + labs(caption = element_blank()) + 
  stat_summary(aes(label = round(..y.., 3)), fun = median, geom = 'label',
               position = position_dodge(width = 1), alpha = 1, size = 7) +
  theme_bw() + theme(text = element_text(size=18)) + easy_remove_legend() + 
  scale_y_continuous(breaks = c(0,0.5,1.0)) + ylim(c(0,1.0)) + 
  labs(x = element_blank(), caption = element_blank(), y = 'Explainable Variance Explained')


graph1 <- graph

vessel_labels2 <- reverse_labels(c(are = 'Architecture (Ext)', ari = 'Architecture (Int)', 
                                  art = 'Art', fac = 'Faces', lsc = 'Landscapes', Combo = 'Combo'))
graph <- reg_results_max %>% left_join(splithalf_summary) %>%
  group_by(train_type, measurement, image_type, dataset) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(!image_type %in% c('Combo'), dataset == 'vessel', 
         train_type %in% c('imagenet'), measurement == 'Beauty') %>%
  mutate(image_type = fct_recode(image_type, !!!vessel_labels2)) %>%
  ggbetweenstats(x = image_type, y = explained_variance, type = 'np',
                 results.subtitle = FALSE, centrality.plotting = FALSE,
                 pairwise.comparisons = FALSE)
graph$layers[[5]]$aes_params$textsize = 5

graph <- graph + labs(caption = element_blank()) + 
  stat_summary(aes(label = round(..y.., 3)), fun = median, geom = 'label',
               position = position_dodge(width = 1), alpha = 1, size = 7) +
  theme_bw() + theme(text = element_text(size=18)) + easy_remove_legend() + 
  scale_y_continuous(breaks = c(0,0.5,1.0)) + ylim(c(0,1.0)) + 
  labs(x = element_blank(), caption = element_blank(), y = 'Explainable Variance Explained')


graph2 <- graph

plot_grid(graph1, graph2 + ylab(element_blank()) + easy_remove_y_axis())



