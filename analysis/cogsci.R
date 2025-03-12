
#### Load packages ####
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
response_data = read.csv('../data/responses_processed.csv') %>% select(-X)
response_data = response_data %>%
  mutate(condition=factor(condition, 
                          levels=c('high', 'medium', 'low'), 
                          labels=c('High', 'Medium', 'Low')))

#### Demographics ####

reportStats <- function(vec, digits=2) {
  return(paste0(round(mean(vec, na.rm = TRUE), digits), '\\pm', 
                round(sd(vec, na.rm = TRUE), digits)))
}
reportStats(subject_data$sex=='female')
reportStats(subject_data$age, 0)
reportStats(subject_data$duration/60, 0)

#### End of demographics ####

#### Plots ####

## Total Points
subject_data$total_points_log = log(subject_data$total_points+1)
plt_points <- ggplot(subject_data, aes(x = condition, y = total_points_log, fill = condition)) +
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
  labs(title = "Total Points Per Condition",
       x = "Rule Compressibility", y = "Total Points (log)") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_points


## Highest Levels
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
levels_per_condition = level_data %>%
  group_by(condition, id) %>%
  summarise(highest_level = max(highest_level)) %>%
  ungroup()
plt_levels = ggplot(levels_per_condition, aes(x = condition, y = highest_level, fill = condition)) +
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
  labs(title = "Highest Levels Per Condition",
       x = "Rule Compressibility", y = "Highest Levels") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_levels


## Points per action
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
  labs(title = "Cumulative Total Points",
       x = "Action ID", y = "Mean Total Points (log)") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_cum_points


## Levels per action
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
  labs(title = "Highest Levels",
      x = "Action ID", y = "Mean Highest Levels") +
  theme(legend.position = "bottom", axis.text = element_text(size = 10))
plt_cum_levels


(plt_points | plt_levels) / (plt_cum_points | plt_cum_levels) +
  plot_annotation(tag_levels = 'a') &
  theme(plot.tag = element_text(size = 16, face = "bold"))
ggsave('plots/beh_performance.png', height = 8, width = 8)


## NAs
plt_nas = ggplot(response_data, aes(x = condition, y = n_NAs, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 1) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black",
    position = position_nudge(x = 0.2), aes(fill = condition), alpha = 0.4
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  geom_point(position = position_jitter(width = 0.1, height = 0), alpha = 0.8, size = 1) + 
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Reported Uncertainty",
       x = "Compressibility", y = "N sentences") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_nas

## Rules
# Exclude outliers
identify_outliers <- function(data_column) {
  Q1 <- quantile(data_column, 0.25)
  Q3 <- quantile(data_column, 0.75)
  IQR_value <- IQR(data_column)
  
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  
  # Mark outliers as TRUE if they are outside the bounds
  outliers <- data_column < lower_bound | data_column > upper_bound
  return(outliers)
}

response_data_filtered = response_data %>%
  filter(len_rules > 0) %>%
  group_by(condition) %>%
  mutate(outlier_len_rules = identify_outliers(len_rules)) %>%
  ungroup() %>%
  filter(!outlier_len_rules) 

plt_rules <- ggplot(response_data_filtered, aes(x = condition, y = len_rules, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 1) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", 
    position = position_nudge(x = 0.2), aes(fill = condition), alpha = 0.4
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  theme_minimal() +
  geom_point(position = position_jitter(width = 0.1, height = 0), alpha = 0.8, size = 1) + 
  scale_fill_manual(values=l_colors)+
  theme_minimal() +
  labs(title = "Reported Rule Lengths",
       x = "Compressibility", y = "Nchar") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
plt_rules

(plt_rules | plt_nas ) + 
  plot_annotation(tag_levels = 'a') &
  theme(plot.tag = element_text(size = 16, face = "bold"))
ggsave('plots/beh_reports.png', height = 4, width = 8)


## Relationships between rule lengths and total points
response_data_filtered$total_points_log = log(response_data_filtered$total_points+1)
ggplot(response_data_filtered, aes(x=len_rules, y=total_points_log, color=condition)) +
  geom_point(size = 3) + 
  geom_smooth(method = 'lm', size = 1.5, aes(color=condition)) +
  scale_color_manual(values = l_colors) +
  theme_minimal() +
  labs(
    title = "Relationship between Reported Rule Length and Performance",
    x = "Reported Rule Length",
    y = "Total Points (Log)",
    color = "Condition"
  ) +
  theme(legend.position = 'none') +
  facet_wrap(~ condition, scales = "free")
ggsave('plots/beh_relations.png', height = 3, width = 8)

#### End of plots ####


#### Stats ####
eta_squared_to_d <- function(eta_squared) {
  d <- sqrt(eta_squared) / sqrt(1 - eta_squared)
  return(d)
}

## Total points per condition

anova_result <- aov(total_points_log ~ condition, data = subject_data)
summary(anova_result)
eta_squared(anova_result)

action_data$condition <- relevel(action_data$condition, ref = "Low")
lme_results <- lmer(total_points_log ~ action_id * condition + (1 | id), data = action_data)
summary(lme_results)

## Levels per condition

anova_result <- aov(highest_level ~ condition, data = levels_per_condition)
summary(anova_result)
eta_squared(anova_result) 

level_data$condition <- relevel(level_data$condition, ref = "Low")
lme_results <- lmer(highest_level ~ action_id * condition + (1 | id), data = level_data)
summary(lme_results)

## Rule lengths per condition

anova_result <- aov(n_rules ~ condition, data = response_data)
summary(anova_result)
eta_squared(anova_result) 

anova_result <- aov(len_rules ~ condition, data = response_data)
summary(anova_result)
eta_squared(anova_result) 

anova_result <- aov(len_rules ~ condition, data = response_data_filtered)
summary(anova_result)
eta_squared(anova_result) 

identify_outliers <- function(data_column) {
  Q1 <- quantile(data_column, 0.25)
  Q3 <- quantile(data_column, 0.75)
  IQR_value <- IQR(data_column)
  
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  
  # Mark outliers as TRUE if they are outside the bounds
  outliers <- data_column < lower_bound | data_column > upper_bound
  return(outliers)
}

# Apply the function to 'len_tips' and 'len_rules'
response_data_filtered = response_data %>%
  filter(len_rules > 0) %>%
  group_by(condition) %>%
  mutate(outlier_len_rules = identify_outliers(len_rules)) %>%
  ungroup() %>%
  filter(!outlier_len_rules) 

anova_result <- aov(len_rules ~ condition, data = response_data_filtered)
summary(anova_result)
eta_squared(anova_result) 


## NAs per condition

anova_result <- aov(n_NAs ~ condition, data = response_data)
summary(anova_result)
eta_squared(anova_result)

anova_result <- aov(len_NAs ~ condition, data = response_data)
summary(anova_result)
eta_squared(anova_result)



## Rule lengths and performance
response_data_filtered$total_points_log = log(response_data_filtered$total_points+1)

cond_high = response_data_filtered %>% filter(condition=='High')
lm_results = lm(total_points_log ~ len_rules, data = cond_high)
summary(lm_results)

cond_low = response_data_filtered %>% filter(condition=='Low')
lm_results = lm(total_points_log ~ len_rules, data = cond_low)
summary(lm_results)

cond_med = response_data_filtered %>% filter(condition=='Medium')
lm_results = lm(total_points_log ~ len_rules, data = cond_med)
summary(lm_results)

lm_results_interaction <- lm(len_rules ~ total_points_log : condition, data = response_data_filtered)
summary(lm_results_interaction)


#### End of stats ####



