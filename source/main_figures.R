# only to be used in conjunction with arxiv_results.R

pacman::p_load('latex2exp','ggtext','ggeasy','viridis','scales','cowplot')

plot_list <- list()
theme_set(theme_bw())

# Levels, Labels, Themes ------------------------------------------------------

train_type_labels <- c('Category-Supervised Models' = 'imagenet', 
                       'Self-Supervised Models' = 'selfsupervised',
                       'Untrained Models' = 'random', 
                       'Humans (Mean-Minus-One)' = 'human')

train_type_levels <- c('Humans (Mean-Minus-One)', 'Category-Supervised Models', 
                       'Self-Supervised Models', 'Untrained Models')

measurement_colors <- c('#3C3184FF', '#4686FBFF', '#80FF53FF') %>%
  set_names(c('Arousal', 'Valence', 'Beauty'))

train_type_colors <- c('#000004FF', '#57157EFF', '#C43C75FF', '#FE9F6DFF') %>%
  set_names(train_type_levels)

# Figure 2: Accuracy ------------------------------------------------------

#* Figure 2A: Overall Scores --------------------------------------------------

mmo_corrs %>% mutate(score = corr, model = as.character(subject), train_type = 'human') %>%
  bind_rows(reg_results_max) %>% 
  filter(dataset == 'Oasis', image_type == 'Combo', 
         train_type %in% train_type_labels) %>%
  mutate(train_type = fct_recode(train_type, !!!train_type_labels),
         train_type = factor(train_type, levels = train_type_levels)) %>%
  ggplot(aes(x = image_type, y = score, color = train_type)) + 
    facet_wrap(~factor(measurement, measurement_levels)) +
    geom_point(alpha = 0.3, position = position_jitterdodge(dodge.width=0.9)) +
    geom_boxplot(position = position_dodge(width=0.9), 
                 outlier.shape = NA, width = 0.5) +
    stat_summary(aes(fill = train_type), geom = 'crossbar',
                 fun.data = median_cl_boot,
                 position = position_dodge(width=0.9), 
                 width = 0.5, lty = 1, alpha = 0.3) +
    geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), 
              alpha = 0.3, fill = 'black', inherit.aes = FALSE,
              data = reliability_combo %>% 
                filter(reliability == 'Split-Half') %>%
                filter(dataset == 'Oasis', image_type == 'Combo')) +
    geom_hline(yintercept = 0, lty = 2) +
    labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
         x = element_blank(), fill = element_blank(), color = element_blank()) +
    scale_fill_manual(values = train_type_colors) + 
    scale_color_manual(values = train_type_colors) + 
    guides(fill=guide_legend(nrow=1, byrow=TRUE)) +
  easy_remove_x_axis() + easy_move_legend('bottom') +
    theme(text = element_text(size = 24),
          legend.text = element_text(size=20, hjust=0.5),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.title.y = element_markdown(size = 20))


#* Figure 2B: Scores across Layers --------------------------------------------------

reg_results %>% filter(dataset == 'Oasis', image_type == 'Combo') %>% 
  mutate(train_type = fct_recode(train_type, !!!train_type_labels[1:3]),
         train_type = factor(train_type, levels = train_type_levels[1:3])) %>%
  filter(str_detect(train_type, 'Category|Self')) %>%
  mutate(depth_binned = cut(model_layer_depth, breaks=seq(0,1.0,0.1), labels=seq(0.1,1.0,0.1))) %>%
  group_by(model, train_type, measurement, image_type, depth_binned) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  ggplot(aes(x = depth_binned, y = score)) +
    facet_wrap(~factor(measurement, measurement_levels)) +
    geom_jitter(aes(color = train_type), alpha = 0.15, width = 0.25) +
    geom_boxplot(outlier.shape = NA, width = 0.5) +
    stat_summary(fun.data = median_cl_boot, geom = 'crossbar',
                 width = 0.5, lty = 1, alpha = 0.3) +
    geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = lower, ymax = upper), 
              alpha = 0.3, fill = 'black', data = reliability_combo %>% 
                filter(dataset == 'oasis', image_type == 'Combo', 
                       reliability == 'Split-Half'), inherit.aes = FALSE) +
    labs(x = 'Relative Layer Depth (Binned)',
         y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)') +
    scale_color_manual(values = train_type_colors) +
    guides(fill = 'none', color = 'none') + easy_move_legend('bottom') +
    theme(text = element_text(size = 24), 
          legend.text = element_text(size=24, hjust=0.5),
          axis.title.y = element_markdown(size=20),
          #axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0)),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank())


# Figure 3: Robustness --------------------------------------------------------

#* Figure 2A: Results by Category ---------------------------------------------

my_breaks <- function(x) { if (max(x) < 5) c(1,2,3,4) else c(5,6,7,8,9)}
my_labels <- function(x) { if (max(x) < 5) c('Animal','Person','Object','Scene') else 
  c('Art','Buildings','Interiors','Faces','Landscapes')}

measurement_plus_levels <- c('Oasis - Arousal', 'Oasis - Valence', 
                             'Oasis - Beauty', 'Vessel - Beauty')

reg_results_max %>% filter(image_type != 'Combo') %>%
  mutate(image_type_numeric = as.numeric(image_type)) %>% 
  filter(train_type %in% c('imagenet','selfsupervised','imagenet21k')) %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  ggplot(aes(x = image_type_numeric, y = score, color = measurement)) +
  facet_wrap(~factor(measurement_plus, measurement_plus_levels), scales = 'free_x', nrow = 1) +
  geom_point(alpha = 0.3, position = position_jitterdodge(jitter.width = 0.3, dodge.width=0.9)) +
  geom_boxplot(aes(group = interaction(measurement,image_type)), 
               position = position_dodge(width=0.9), outlier.shape = NA, width = 0.5) +
  stat_summary(aes(fill = measurement), geom = 'crossbar', fun.data = median_cl_boot, fill = 'white', 
               position = position_dodge(width=0.9), width = 0.5, lty = 1, alpha = 0.3) +
  geom_rect(aes(xmin = image_type_numeric-0.3, xmax = image_type_numeric+0.3, 
                ymin = lower, ymax = upper, fill = measurement), alpha = 0.3,
            position = position_dodge(width = 0.9), show.legend = FALSE,
            data = reliability_combo %>% filter(image_type != 'Combo') %>%
              mutate(image_type_numeric = as.numeric(image_type)) %>%
              mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
              filter(reliability == 'Split-Half'), inherit.aes = FALSE) +
  scale_x_continuous(breaks = my_breaks, labels = my_labels) + custom_themes$bottom_legend +
  labs(y = "*r<sub>Pearson</sub>*(Predicted, Actual Ratings)",
       x = element_blank(), fill = element_blank(), color = element_blank()) +
  easy_rotate_x_labels(angle = 25, side = 'right') + easy_remove_legend() +
  scale_fill_manual(values = rep(NA, 4)) + 
  scale_color_manual(values = rep('black', 4)) +
  theme(text = element_text(size = 24),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_markdown(size = 18))

#* Figure 2B: Subject Analysis --------------------------------------------------

subject_regs %>% group_by(subject, dataset, measurement, image_type) %>%
  summarise(score = mean(score, na.rm = TRUE)) %>%
  filter(dataset == 'Oasis', image_type == 'Combo') %>%
  mutate(statistic = 'Average') %>%
  ggplot(aes(x = statistic, y = score)) +
  facet_wrap(~ measurement) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  geom_boxplot(position = position_dodge(width=0.9), 
               outlier.shape = NA, width = 0.5) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar',
               width = 0.5, lty = 1, fill = 'black', alpha = 0.3) + 
  custom_themes$bottom_legend + #easy_remove_legend() +
  labs(y = '*r<sub>Pearson</sub>*(Predicted, Actual Ratings)',
       #x = 'Model Selection', 
       x = element_blank(),
       fill = element_blank(), color = element_blank()) +
  scale_y_continuous(limits = c(-0.25,0.85), 
                     breaks = seq(-0.25,0.75,0.25)) +
  theme(text = element_text(size = 24),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.y = element_markdown(size = 24))

#* Figure 2C: Cross-Decoding --------------------------------------------------

cross_decoding_labels <- c('Object' = 'O: Object',  'Animal' = 'O: Animal', 
                           'Person' = 'O: Person', 'Scene' = 'O: Scene',
                           'art' = 'V: Art', 'fac' = 'V: Faces', 'lsc' = 'V: Landscapes', 
                           'are' = 'V: Buildings', 'ari' = 'V: Interiors', 
                           'oasis_combo' = 'O: Combo', 'vessel_combo' = 'V: Combo')

cross_decode_regex <- 'Scene|Landscapes|Faces|Buildings|Interiors|Art'
cross_decode_zoom <- c('Art','Faces','Buildings','Interiors','Landscapes','Scene')

cross_decoding_overall %>% select(model, train_type, train_data, test_data, cross_score) %>%
  mutate(train_data = fct_recode(factor(train_data), !!!reverse_labels(cross_decoding_labels)),
         test_data = fct_recode(factor(test_data), !!!reverse_labels(cross_decoding_labels))) %>%
  filter(train_data != test_data, !str_detect(train_type, 'clip|slip')) %>%
  mutate(train_data = str_replace(train_data, 'V: |O: ', ''),
         test_data = str_replace(test_data, 'V: |O: ', '')) %>%
  filter(train_data %in% cross_decode_zoom, test_data %in% cross_decode_zoom) %>%
  group_by(model, train_type, train_data, test_data) %>% 
  mutate(score = cross_score) %>% summarise(score = mean(score)) %>%
  mutate(scoring = 'cross_decoding') %>%
  bind_rows(reg_results_max %>% filter(measurement == 'Beauty') %>%
              filter(train_type %in% c('imagenet','selfsupervised')) %>%
              filter(image_type %in% cross_decode_zoom) %>%
              mutate(train_data = image_type, test_data = image_type) %>%
              select(model, train_type, train_data, test_data, score) %>% 
              mutate(scoring = 'loocv')) %>%
  group_by(train_data, test_data) %>%
  summarise(n = n(), ci = list(mean_cl_boot(score))) %>% 
  unnest(ci) %>% mutate_at(vars(y, ymin, ymax), round, 2) %>%
  mutate(report = paste0(y,'\n [',ymin,', ',ymax,']')) %>%
  mutate(train_data = factor(train_data, levels = cross_decode_zoom),
         test_data = factor(test_data, levels = cross_decode_zoom)) %>%
  ggplot(aes(x = train_data, y = test_data, fill = y)) +
  geom_tile() + theme_minimal() +
  scale_x_discrete(expand=c(0,0), position = 'bottom') +
  scale_y_discrete(expand=c(0,0), position = 'left') +
  geom_text(aes(label = report), size = 9) +
  scale_fill_gradient2(low="#A63446", mid = 'white', high="#0C6291", limits=c(-1,1)) +
  #labs(y = 'Train On', x = '\nTest On', fill = '') +
  labs(x = element_blank(), y = element_blank(), fill=element_blank()) +
  #easy_rotate_y_labels(angle = 90, side = 'middle') +
  guides(fill=guide_colorbar(position='bottom', direction='horizontal')) +
  theme(legend.key.width = unit(3, 'cm'), legend.key.height = unit(1, 'cm'),
        text=element_text(size = 30))

# Figure 4: Language-Alignment ------------------------------------------------

model_subset <- c('ViT-L/14','RN50','convnext_large_in22k','swin_base_patch4_window7_224_in22k',
                  'resnetv2_101x3_bitm_in21k',
                  'RegNet-64Gf-SEER','swsl_resnet50','dino_xcit_small_12_p16')
model_labels <- c('Best CLIP Model (Vision Transformer - L14)', 'Worst CLIP Model (ResNet50x4)', 
                  'Best ImageNet21K (Supervised) Model', 'Best ImageNet (Supervised) Model',
                  'BiTm (300M-Image-Soft-Supervised Model)', 'SEER (1B-Image-Self-Supervised Model)',
                  '900M-Image-Semi-Supervised Model', 'Best ImageNet (Self-Supervised) Model')

names(model_subset) <- model_labels

plot_list$clip_results = list()

plot_list$clip_results[[1]] <- boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(model %in% model_subset) %>%
  filter(image_type == 'Combo') %>%
  group_by(model, train_type, dataset, measurement, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(model = factor(model, levels=model_subset)) %>%
  mutate(model = fct_recode(model, !!!model_subset)) %>%
  ggplot(aes(x = reorder(model, -explained_variance), y = explained_variance)) + 
  geom_col(width = 1, fill = 'black', color = 'black', alpha = 0.3) + 
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), 
                width = 1, color = 'black', alpha = 0.3) +
  theme(text = element_text(size = 30)) + 
  geom_text(aes(y = 0.05, x = model, label = model), size = 7, hjust = 0, check_overlap = TRUE) + 
  labs(y = '% Explainable Variance Explained', x = '') + 
  scale_x_discrete(labels = rev(model_labels), limits = rev, expand = c(0.13,0)) + coord_flip() +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0,1.0,0.1), limits = c(0,1.0))

plot_list$clip_results[[2]] <- boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(train_type == 'slip', !str_detect(model, 'Ep100|CC12M')) %>%
  group_by(model, train_type, bootstrap_id) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(slip_kind = str_split(model, '-', simplify = TRUE)[, 3]) %>%
  mutate(slip_size = str_split(model, '-', simplify = TRUE)[, 2],
         slip_size = fct_recode(factor(slip_size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  mutate(slip_kind = factor(slip_kind, levels = c('SLIP','CLIP','SimCLR'))) %>%
  mutate(model = paste0('ViT-',slip_size)) %>%
  mutate(model = factor(model, levels = c('ViT-Small','ViT-Base','ViT-Large'))) %>%
  ggplot(aes(x = model, y = explained_variance, fill = slip_kind)) + 
  geom_col(color = 'black', position = position_dodge(width = 0.9),
           width = 0.9, lty = 1, alpha = 0.4) +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
               position = position_dodge(width = 0.9),
               width = 0.9, lty = 1, alpha = 0.3) +
  geom_text(aes(y = 0.05, x = model, label = model), 
            size = 7, hjust = 0, check_overlap = TRUE) + 
  theme(text = element_text(size = 30)) + custom_themes$bottom_legend +
  labs(y = '% Explainable Variance Explained', x = '', fill = '') + coord_flip() +
  scale_fill_manual(values = plasma(n = 3, direction = -1, begin = 0.2, end = 0.8)) +
  easy_remove_y_axis() + scale_y_continuous(breaks = seq(0,1.0,0.1), limits = c(0,1.0))

cowplot::plot_grid(plot_list$clip_results[[1]] + easy_remove_x_axis(what = 'title') + 
                     scale_y_continuous(breaks = seq(0,0.9,0.15), limits = c(0,0.9)),
                   NULL, plot_list$clip_results[[2]] + 
                     easy_remove_x_axis(what = 'title') + easy_remove_legend() +
                     scale_y_continuous(breaks = seq(0,0.75,0.15), limits = c(0,0.8)),
                   rel_widths = c(0.6,0.05,0.35), nrow = 1)
  
boot_results %>% left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  filter(image_type == 'Combo') %>%
  filter(model == 'ViT-L/14' | str_detect(train_type, 'slip'), 
         !str_detect(model, 'Ep100|CC12M')) %>%
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
  group_by(model, train_type, bootstrap_id, measurement_plus) %>%
  summarise(explained_variance = mean(explained_variance)) %>%
  group_by(model, train_type, measurement_plus) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            explained_variance = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(slip_kind = str_split(model, '-', simplify = TRUE)[, 3]) %>%
  mutate(slip_kind = ifelse(slip_kind == '', 'OpenAI-CLIP', slip_kind)) %>%
  mutate(slip_size = str_split(model, '-', simplify = TRUE)[, 2]) %>%
  mutate(slip_size = ifelse(slip_size == 'L/14', 'Large-Custom', slip_size)) %>%
  mutate(slip_size = fct_recode(factor(slip_size), 'Base'='B', 'Small'='S', 'Large'='L')) %>%
  mutate(slip_kind = factor(slip_kind, levels = c('OpenAI-CLIP', 'SLIP','CLIP','SimCLR'))) %>%
  mutate(model = paste0('ViT-',slip_size)) %>%
  mutate(model = factor(model, levels = c('ViT-Small','ViT-Base','ViT-Large', 'ViT-Large-Custom'))) %>%
  ggplot(aes(x = model, y = explained_variance, fill = slip_kind)) + 
    facet_wrap(~measurement_plus, nrow = 1) +
    geom_col(color = 'black', position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
             width = 0.9, lty = 1, alpha = 0.4) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), color = 'black',
                  position = position_dodge2(width = 0.9, preserve = 'single', padding = 0),
                  width = 0.9, lty = 1, alpha = 0.3) +
    geom_text(aes(y = 0.05, x = model, label = model), 
              size = 5, hjust = 0, check_overlap = TRUE) + 
    labs(y = '% Explainable Variance Explained', x = '', fill = '') + coord_flip() +
    scale_fill_manual(values = c('darkgrey', plasma(n = 3, direction = -1, begin = 0.2, end = 0.8))) +
    scale_y_continuous(breaks = seq(0.1,0.9,0.2), limits = c(0,1.0)) +
    easy_remove_y_axis() + easy_move_legend('bottom') +
    theme(text = element_text(size = 24),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.spacing.x = unit(0.5, "lines"))


# Supplementary Figures -------------------------------------------------------

##* Taskonomy Rankings --------------------------------------------------------

boot_results %>% filter(train_type == 'taskonomy') %>%
  filter(dataset == 'Oasis', image_type %in% c('Combo')) %>%
  left_join(splithalf_summary) %>%
  mutate(explained_variance = (score**2 / r**2)) %>%
  group_by(model, train_type, measurement, image_type) %>%
  summarise(lower_ci = quantile(explained_variance, 0.025, na.rm = TRUE),
            upper_ci = quantile(explained_variance, 0.975, na.rm = TRUE),
            score = mean(explained_variance, na.rm = TRUE)) %>% ungroup() %>%
  mutate(rank = sprintf("%04i", as.integer(rank(score)))) %>%
  left_join(model_display_names) %>%
  mutate(model_display_name = str_replace(model_display_name, 'Unsupervised', '')) %>%
    {ggplot(., aes(rank, score)) + facet_wrap(~measurement, scales = 'free') +
        geom_bar(stat = 'identity', position = 'identity') + 
        geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.1) +
        scale_x_discrete(labels = with(., model_display_name %>% set_names(rank))) +
        labs(x = '', y = "Explainable Variance Explained") +
        scale_y_continuous(breaks = seq(0,0.3,0.1)) +
        coord_flip(ylim = c(-0.05,0.35), clip = 'on')} + theme_bw() +
    #theme(text = element_text(size=24), axis.title.x = element_markdown()) 
        theme(text = element_text(size=24),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              axis.title.x = element_markdown())

##* Language Gains in Focus ---------------------------------------------------

x_breaks <- function(x) { if (max(x) < 6) c(1,2,3,4,5) else c(5,6,7,8,9,10)}
x_labels <- function(x) { if (max(x) < 6) c('Animal','Person','Object','Scene', 'Combo') else 
  c('Art','Buildings','Interiors','Faces','Landscapes', 'Combo')}

type_labels <- c('Top-Ranking Language-Aligned Model',
                 'Top-Ranking Unimodal Vision Model')

boot_results %>% 
  mutate(language = ifelse(str_detect(train_type, 'clip'), type_labels[[1]], type_labels[[2]])) %>%
  select(model, train_type, bootstrap_id, dataset, measurement, 
         image_type, language, score) %>%
  group_by(bootstrap_id, dataset, measurement, image_type, language) %>%
  mutate(rank = dense_rank(-score)) %>% filter(rank == 1) %>%
  group_by(dataset, measurement, image_type, language) %>%
  summarise(lower_ci = quantile(score, 0.025, na.rm = TRUE),
            upper_ci = quantile(score, 0.975, na.rm = TRUE),
            explained_variance = mean(score, na.rm = TRUE)) %>% ungroup() %>%
  mutate(image_type_numeric = as.numeric(image_type)) %>% 
  mutate(image_type_numeric = ifelse(dataset == 'Oasis' & image_type == 'Combo', 
                                     5, image_type_numeric)) %>% 
  mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
    ggplot(aes(x = image_type_numeric, y = explained_variance, group = language)) + 
    geom_col(aes(fill = language), position = position_dodge(width = 0.9)) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.3, 
                  position = position_dodge(width = 0.9)) +
    facet_wrap(~measurement_plus, scales = 'free_x', ncol = 4) +
    geom_rect(aes(xmin = image_type_numeric-0.45, xmax = image_type_numeric+0.45, 
                  ymin = lower, ymax = upper), alpha = 0.3, 
              position = position_dodge(width = 0.9), show.legend = FALSE,
              data = reliability_combo %>% #filter(image_type != 'Combo') %>%
                mutate(image_type_numeric = as.numeric(image_type)) %>%
                mutate(image_type_numeric = ifelse(dataset == 'Oasis' & image_type == 'Combo',
                                                   5, image_type_numeric)) %>%
                mutate(measurement_plus = paste(dataset, measurement, sep = ' - ')) %>%
                filter(reliability == 'Split-Half'), inherit.aes = FALSE) +
    labs(y = "*r<sub>Pearson</sub>*(Predicted, Actual Ratings)",
         x = element_blank(), fill = element_blank(), color = element_blank()) +
    #guides(fill=guide_legend(nrow=2, byrow=TRUE)) +
    scale_fill_manual(values = c('#4199FF','lightgray')) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels) +
    theme_bw() + easy_legend_at('bottom') +
    easy_rotate_x_labels(angle = 25, side = 'right') +
    theme(text = element_text(size=24),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.justification="center", 
          legend.box.margin=margin(-12,0,0,0),
          axis.title.y = element_markdown())

rstudioapi::executeCommand("exportPlot")
  


  