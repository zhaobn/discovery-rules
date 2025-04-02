#### Load packages ####
options(scipen=999)

library(dplyr)
library(tidyr)
library(ggpubr)
library(lme4)
library(lmerTest)
library(effectsize)
library(purrr)

library(ggplot2)
library(gghalves)
library(patchwork)
library(MoMAColors)
l_colors = moma.colors("Levine1", 3)


#### Load data ####
subject_data = read.csv('../data/subject_data.csv')
subject_data = subject_data %>%
  mutate(condition=factor(condition, 
                          levels=c('high', 'medium', 'low'), 
                          labels=c('High', 'Medium', 'Low')))
action_data = read.csv('../data/action_data.csv') %>% select(-X)  %>%
  mutate(condition=factor(condition, 
                          levels=c('high', 'medium', 'low'), 
                          labels=c('High', 'Medium', 'Low')))

# use code in cogsci.R for existing plots:
# - plt_points, plt_levels, plt_cum_points, plt_cum_levels


#### Rate of exploration ####

# Overall combination success
combine_rate_per_condition = action_data %>%
  mutate(is_combine=action=='combine') %>%
  group_by(condition, id) %>%
  summarise(combine_rate = sum(is_combine)/n()) %>%
  ungroup()
plt_combine_rate = ggplot(combine_rate_per_condition, aes(x = condition, y = combine_rate, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 1) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", aes(fill=condition), 
    position = position_nudge(x = 0.2), alpha = 0.4
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  geom_point(position = position_jitter(width = 0.1, height = 0), alpha = 0.8, size = 2) +
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Proportion of Combine Action Per Condition",
       x = "Rule Compressibility", y = "Proportion") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_combine_rate
  

combine_success_per_condition = action_data %>%
  filter(action=='combine') %>%
  mutate(success = nchar(yield) > 0) %>%
  group_by(condition, id) %>%
  summarise(combine_success = sum(success)/n()) %>%
  ungroup()
plt_combine_success = ggplot(combine_success_per_condition, aes(x = condition, y = combine_success, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 1) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", aes(fill=condition), 
    position = position_nudge(x = 0.2), alpha = 0.4
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  geom_point(position = position_jitter(width = 0.1, height = 0), alpha = 0.8, size = 2) +
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Success Rate of Combine Action Per Condition",
       x = "Rule Compressibility", y = "Success Rate") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_combine_success



# combine proportion per step
combine_data_summary = action_data %>%
  mutate(is_combine=action=='combine') %>%
  group_by(condition, action_id) %>%
  summarise(
    mean_combine_rate = mean(is_combine, na.rm = TRUE),
    se_combine_rate = sd(is_combine, na.rm = TRUE)/sqrt(n())
  )
plt_combine_per_step = ggplot(combine_data_summary, aes(x = action_id, y = mean_combine_rate, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_combine_rate - se_combine_rate, 
                  ymax = mean_combine_rate + se_combine_rate, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 2) + 
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Proportion of Combine Action Per Step",
       x = "Action ID", y = "Mean Proportion of Combine Action") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_combine_per_step


combine_success_summary = action_data %>%
  filter(action=='combine') %>%
  mutate(is_success=nchar(yield)>0) %>%
  group_by(condition, action_id) %>%
  summarise(
    mean_combine_success_rate = mean(is_success, na.rm = TRUE),
    se_combine_success_rate = sd(is_success, na.rm = TRUE)/sqrt(n())
  )
plt_combine_success_per_step = ggplot(combine_success_summary, aes(x = action_id, y = mean_combine_success_rate, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_combine_success_rate - se_combine_success_rate, 
                  ymax = mean_combine_success_rate + se_combine_success_rate, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 2) + 
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Combine Success Rate Per Step",
       x = "Action ID", y = "Mean Combine Success Rate") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_combine_success_per_step



# diff from prev combine attempt
combine_feature_data = action_data %>%
  filter(action=='combine') %>%
  mutate(
    target_shape = sub("_.*", "", target),
    target_texture = sub(".*?_(.*?)_.*", "\\1", target),
    target_level = sub(".*_", "", target),
    held_shape = sub("_.*", "", held),
    held_texture = sub(".*?_(.*?)_.*", "\\1", held),
    held_level = sub(".*_", "", held)
  ) %>%
  select(condition, id, action_id, held_shape, held_texture, held_level,
         target_shape, target_texture, target_level, yield, points, total_points, total_points_log, held, target) %>%
  arrange(id, action_id) %>%
  group_by(id) %>%
  mutate(combine_id = row_number()) %>%
  ungroup()

combine_feature_A = combine_feature_data %>%
  select(id, combine_id, target_shape, target_texture, held_shape, held_texture)
combine_feature_B = combine_feature_data %>%
  filter(combine_id>1) %>%
  mutate(combine_id = combine_id-1) %>%
  select(id, combine_id, target_shape, target_texture, held_shape, held_texture)
combine_feature_joined = combine_feature_B %>%
  left_join(combine_feature_A, by=c('id', 'combine_id'))
combine_feature_joined_cal = combine_feature_joined %>%
  mutate(
    target_shape_diff = as.integer(target_shape.x != target_shape.y),
    target_texture_diff = as.integer(target_texture.x != target_texture.y),
    held_shape_diff = as.integer(held_shape.x != held_shape.y),
    held_texture_diff = as.integer(held_texture.x != held_texture.y)
  ) %>%
  mutate(feat_diff = target_shape_diff + target_texture_diff + held_shape_diff + held_texture_diff) %>%
  mutate(combine_id = combine_id + 1) %>%
  select(id, combine_id, feat_diff)

combine_feature_check = combine_feature_data %>%
  left_join(combine_feature_joined_cal, by=c('id', 'combine_id')) %>%
  select(id, combine_id, target, held, feat_diff, condition) %>%
  filter(combine_id > 1)


combine_feature_overall = combine_feature_check %>%
  group_by(condition, id) %>%
  summarise(feat_diff = sum(feat_diff)/n()) %>%
  ungroup()
plt_diffs <- ggplot(combine_feature_overall, aes(x = condition, y = feat_diff, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 1) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", aes(fill = condition),
    position = position_nudge(x = 0.2), alpha=0.4
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  geom_point(position = position_jitter(width = 0.1, height = 0), alpha = 0.8, size = 2) +
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Feature Diffs Per Condition",
       x = "Rule Compressibility", y = "Feature Diffs") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_diffs

combine_feature_summary = combine_feature_check %>%
  group_by(combine_id, condition) %>%
  summarise(
    mean_diff = mean(feat_diff, na.rm = TRUE),
    se_dff = sd(feat_diff, na.rm = TRUE)/sqrt(n())
  )

plt_combine_diff_per_step = ggplot(combine_feature_summary, aes(x = combine_id, y = mean_diff, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_diff - se_dff, 
                  ymax = mean_diff + se_dff, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 2) + 
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Feature Diff Per Combine Action",
       x = "Combine ID", y = "Mean Feature Diff") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_combine_diff_per_step

