
# Load packages ----
options(scipen=999)

library(dplyr)
library(tidyr)
library(ggpubr)
library(lme4)
library(lmerTest)
library(effectsize)

library(ggplot2)
library(gghalves)
library(patchwork)
library(MoMAColors)
l_colors = moma.colors("Levine1", 3)


# Load data ----
action_data = read.csv('../data/action_data.csv') %>% select(-X)  %>%
  mutate(condition=factor(condition, levels=c('high', 'medium', 'low'), labels=c('High', 'Medium', 'Low')))

base_model = read.csv('../modeling/python/simulation_results/base_online_sim_summary.csv')
base_model$condition = recode(
  base_model$condition,
  "hard" = "low",
  "med" = "medium",
  "simple" = "high"
) 
base_model = base_model %>%
  mutate(condition=factor(condition, levels=c('high', 'medium', 'low'), labels=c('High', 'Medium', 'Low')))

pcfg_model = read.csv('../modeling/python/simulation_results/pcfg_online_sim_summary.csv')
pcfg_model$condition = recode(
  pcfg_model$condition,
  "hard" = "low",
  "med" = "medium",
  "simple" = "high"
) 
pcfg_model = pcfg_model %>%
  mutate(condition=factor(condition, levels=c('high', 'medium', 'low'), labels=c('High', 'Medium', 'Low')))


# Plot rewards ----
level_data = action_data %>%
  select(id, action_id, action, held, target, yield, condition) %>%
  mutate(level = ifelse(
    nchar(yield) > 1 & !is.na(yield),
    as.numeric(sub(".*_(\\d+)$", "\\1", yield)),
    0
  )) %>% 
  group_by(id) %>%
  arrange(action_id) %>%
  mutate(highest_level = cummax(level)) %>%
  ungroup()

action_data = action_data %>%
  group_by(id) %>%
  arrange(action_id) %>%
  mutate(total_points = cumsum(points)) %>%
  mutate(total_points_log = ifelse(total_points < 1, 0, ifelse(total_points == 1, 0.1, log(total_points))))

action_data_summary = action_data %>%
  group_by(action_id, condition) %>%
  summarise(
    mean_total_points_log = mean(total_points_log, na.rm = TRUE),
    se_total_points_log = sd(total_points_log, na.rm = TRUE) / sqrt(n())
  )
plt_cum_points = ggplot(action_data_summary, aes(x = action_id, y = mean_total_points_log, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_total_points_log - se_total_points_log, 
                  ymax = mean_total_points_log + se_total_points_log, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 2) + 
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Participants",
       x = "Action ID", y = "Mean Total Points (log)") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_cum_points

plt_cum_points_base = ggplot(base_model, aes(x=action, y = log_cum_rewards_mean, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = log_cum_rewards_ci_lo,
                  ymax = log_cum_rewards_ci_hi,
                  fill = condition), alpha = 0.2, color = NA) +
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "PSRL (uniform)",
       x = "Action ID", y = "Mean Total Points (log)") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))

plt_cum_points_pcfg = ggplot(pcfg_model, aes(x=action, y = log_cum_rewards_mean, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = log_cum_rewards_ci_lo,
                  ymax = log_cum_rewards_ci_hi,
                  fill = condition), alpha = 0.2, color = NA) +
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "PSRL (structured)",
       x = "Action ID", y = "Mean Total Points (log)") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))

(plt_cum_points_base | plt_cum_points_pcfg | plt_cum_points) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")


level_data_summary = level_data %>%
  group_by(action_id, condition) %>%
  summarise(
    mean_highest_level = mean(highest_level, na.rm = TRUE),
    se_highest_level = sd(highest_level, na.rm = TRUE) / sqrt(n())
  )
plt_cum_levels = ggplot(level_data_summary, aes(x = action_id, y = mean_highest_level, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_highest_level - se_highest_level, 
                  ymax = mean_highest_level + se_highest_level, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 2) + 
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +   
  labs(title = "Participants",
       x = "Action ID", y = "Mean Highest Levels") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_cum_levels

plt_cum_levels_base = ggplot(base_model, aes(x=action, y = highest_levels_mean, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = highest_levels_ci_lo,
                  ymax = highest_levels_ci_hi,
                  fill = condition), alpha = 0.2, color = NA) +
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "PSRL (uniform)",
       x = "Action ID", y = "Mean Highest Levels") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))

plt_cum_levels_pcfg = ggplot(pcfg_model, aes(x=action, y = highest_levels_mean, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = highest_levels_ci_lo,
                  ymax = highest_levels_ci_hi,
                  fill = condition), alpha = 0.2, color = NA) +
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "PSRL (structured)",
       x = "Action ID", y = "Mean Highest Levels") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))

(plt_cum_levels_base | plt_cum_levels_pcfg | plt_cum_levels) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")



# Recovery (models) ----
cover_base = base_model %>%
  select(condition, action, recover_rate_mean, recover_rate_ci_lo, recover_rate_ci_hi) %>%
  mutate(model='uniform')
cover_pcfg = pcfg_model %>%
  select(condition, action, recover_rate_mean, recover_rate_ci_lo, recover_rate_ci_hi) %>%
  mutate(model='structured')
cover_models = rbind(cover_base, cover_pcfg)

ggplot(cover_models, aes(x = action, y = recover_rate_mean, color = condition, linetype = model)) +
  geom_line() +
  geom_ribbon(aes(ymin = recover_rate_ci_lo, ymax = recover_rate_ci_hi, fill = condition, group = interaction(condition, model)),
              alpha = 0.2, color = NA) +
  scale_color_manual(values = l_colors) +
  scale_fill_manual(values = l_colors) +
  #scale_y_continuous(limits = c(0, 1.15)) +
  labs(title = "", x = "Action ID", y = "Mean recover rate") +
  theme_minimal() +
  theme(legend.position = "right", axis.text = element_text(size = 10))




# Explorative ----

cum_combine_success = action_data %>%
  group_by(id) %>%
  arrange(action_id, .by_group = TRUE) %>%
  mutate(
    is_combine = action == "combine",
    is_success = is_combine & yield != "",
    cum_combine = cumsum(is_combine),
    cum_success = cumsum(is_success),
    cum_success_rate = ifelse(cum_combine > 0, cum_success / cum_combine, 0)
  ) %>%
  ungroup()

success_data_summary = cum_combine_success %>%
  group_by(action_id, condition) %>%
  summarise(
    mean_success_rate = mean(cum_success_rate, na.rm = TRUE),
    se_success_rate = sd(cum_success_rate, na.rm = TRUE) / sqrt(n())
  )
ggplot(success_data_summary, aes(x = action_id, y = mean_success_rate, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_success_rate - se_success_rate, 
                  ymax = mean_success_rate + se_success_rate, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 2) + 
  scale_color_manual(values=l_colors)+
  scale_fill_manual(values=l_colors)+
  theme_minimal() +   
  labs(title = "Participants",
       x = "Action ID", y = "Cumulative success rate") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_cum_levels


x = cum_combine_success %>%
  filter(condition=='High') %>%
  filter(action_id == 1)






