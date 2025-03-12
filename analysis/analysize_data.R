library(dplyr)
library(tidyr)
library(ggplot2)
library(gghalves)

library(effectsize)
options(scipen=999)


l_colors = moma.colors("Levine1", 3)
p_colors = moma.colors("Picabia", 3)

subject_data = read.csv('../data/subject_data.csv')
subject_data = subject_data %>%
  #mutate(condition=ifelse(condition=='medium-1', 'medium', condition)) %>%
  mutate(condition=ifelse(condition=='easy', 'high', 
                          ifelse(condition=='hard', 'low', condition))) %>%
  select(id, condition, messageHow, messageRules, total_points, 
         age, sex, engagement, difficulty, feedback, date, time, duration)




#### Demographics ####
subject_data = read.csv('../data/subject_data.csv')
assignment_orders = c('easy', 'medium', 'hard')

subject_data = subject_data %>%
  #mutate(condition=ifelse(condition=='medium-1', 'medium', condition)) %>%
  mutate(condition=factor(condition, levels=assignment_orders)) %>%
  select(id, condition, messageHow, messageRules, total_points, 
         age, sex, engagement, difficulty, feedback, date, time, duration)


reportStats <- function(vec, digits=2) {
  return(paste0(round(mean(vec, na.rm = TRUE), digits), '\\pm', 
                round(sd(vec, na.rm = TRUE), digits)))
}
reportStats(subject_data$age)


#### Fix task duration ####
# prolific_data = read.csv('../data/prolific_export.csv') 
# prolific_starttime = prolific_data %>%
#   select(prolific_id = Participant.id, start_time = Started.at, duration=Time.taken)
#   
# id_data = read.csv('../data/id_data.csv')
# time_data = id_data %>%
#   left_join(prolific_starttime, by='prolific_id') %>%
#   select(id, start_time, duration)
# reportStats(time_data$duration/60)
# 
# subject_data = subject_data %>%
#   left_join(time_data, by = 'id') %>%
#   select(-c('start_time', 'task_duration'))
# write.csv(subject_data, file='../data/subject_data.csv')
reportStats(subject_data$duration/60)

reportStats(subject_data$sex=='female')


# Points per condition
subject_data$total_points_log = log(subject_data$total_points)
ggplot(subject_data, aes(x = condition, y = total_points_log, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 0.6) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
    position = position_nudge(x = 0.2)
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  theme_minimal() +
  labs(title = "Total Points per Condition",
       x = "Condition", y = "Total Points (log)") +
  theme(legend.position = "none")


subject_data$total_points_log <- log(subject_data$total_points + 1)
anova_result <- aov(total_points_log ~ condition, data = subject_data)
summary(anova_result)
shapiro.test(residuals(anova_result)) # passed, W=0.96, p=0.2
eta_squared(anova_result) # 0.19


anova_raw <- aov(total_points ~ condition, data = subject_data)
shapiro.test(residuals(anova_raw))

# points per action
action_data = read.csv('../data/action_data.csv') %>% select(-X)
# action_data = action_data %>%
  # mutate(condition=ifelse(assignment=='easy', 'high',
  #                         ifelse(assignment=='hard', 'low', 'medium'))) %>%
  # select(-assignment) %>%
  # filter(action_id < 41)
# write.csv(action_data, file = '../data/action_data.csv')

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
ggplot(action_data_summary, aes(x = action_id, y = mean_total_points_log, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_total_points_log - se_total_points_log, 
                  ymax = mean_total_points_log + se_total_points_log, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 3) + 
  theme_minimal() +   
  labs(x = "Action ID", y = "Mean Log of Total Points") + 
  theme(legend.position = "bottom")



# highest levels per condition
level_data = action_data %>%
  select(id, action_id, action, held, target, yield, condition) %>%
  mutate(level = ifelse(
    nchar(yield) > 1 & !is.na(yield),
    as.numeric(sub(".*_(\\d+)$", "\\1", yield)),
    0
  ))
level_data <- level_data %>%
  group_by(id) %>%
  arrange(action_id) %>%
  mutate(highest_level = cummax(level)) %>%
  ungroup()


levels_per_condition = level_data %>%
  group_by(condition, id) %>%
  summarise(highest_level = max(highest_level)) %>%
  ungroup()

ggplot(levels_per_condition, aes(x = condition, y = highest_level, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 0.6) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
    position = position_nudge(x = 0.2)
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  theme_minimal() +
  labs(title = "Highest Levels per Condition",
       x = "Condition", y = "Highest Levels") +
  theme(legend.position = "none")

anova_result <- aov(highest_level ~ condition, data = levels_per_condition)
summary(anova_result)
shapiro.test(residuals(anova_result)) # passed, W=0.94, p<0.0001
eta_squared(anova_result) # 0.22


# highest levels per action
level_data_summary = level_data %>%
  group_by(action_id, condition) %>%
  summarise(
    mean_highest_level = mean(highest_level, na.rm = TRUE),
    se_highest_level = sd(highest_level, na.rm = TRUE) / sqrt(n())
  )
ggplot(level_data_summary, aes(x = action_id, y = mean_highest_level, color = condition)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_highest_level - se_highest_level, 
                  ymax = mean_highest_level + se_highest_level, 
                  fill = condition), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = condition), size = 3) + 
  theme_minimal() +   
  labs(x = "Action ID", y = "Mean Highest Levels") + 
  theme(legend.position = "bottom")


##### Message length per condition
subject_data$raw_rule_length = nchar(subject_data$messageRules)
ggplot(subject_data, aes(x = condition, y = raw_rule_length, fill = condition)) +  stat_summary(
  fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
  # position = position_nudge(x = 0.2)
) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    #position = position_nudge(x = 0.2)
  ) +
  geom_jitter(width = 0.15, size = 2, alpha = 0.6, color = "black") + 
  
  theme_minimal() +
  labs(title = "Rules Nchar per Condition",
       x = "Condition", y = "Nchar") +
  theme(legend.position = "none")


subject_data$message_length = nchar(subject_data$messageRules) + nchar(subject_data$messageHow)
ggplot(subject_data, aes(x = condition, y = message_length, fill = condition)) +  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
   # position = position_nudge(x = 0.2)
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    #position = position_nudge(x = 0.2)
  ) +
  geom_jitter(width = 0.15, size = 2, alpha = 0.6, color = "black") + 
  
  theme_minimal() +
  labs(title = "Rules Nchar per Condition",
       x = "Condition", y = "Nchar") +
  theme(legend.position = "none")

anova_result <- aov(message_length ~ condition, data = subject_data)
summary(anova_result)
shapiro.test(residuals(anova_result)) # passed, W=0.96, p=0.2
eta_squared(anova_result) # 0.06




subject_data$message_diff = nchar(subject_data$messageRules) - nchar(subject_data$messageHow)
ggplot(subject_data, aes(x = condition, y = message_diff, fill = condition)) +  stat_summary(
  fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
  # position = position_nudge(x = 0.2)
) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    #position = position_nudge(x = 0.2)
  ) +
  geom_jitter(width = 0.15, size = 2, alpha = 0.6, color = "black") + 
  
  theme_minimal() +
  labs(title = "Rules Nchar per Condition",
       x = "Condition", y = "Nchar") +
  theme(legend.position = "none")






# Points and lengths
ggplot(subject_data, aes(x=total_points_log, y=message_length, color=condition)) +
  geom_point(aes(shape=condition), size=3) +
  geom_smooth(method = "lm", se = TRUE, aes(fill = condition), alpha = 0.1) +
  theme_minimal() +
  labs(title = "Total Points and Message Lengths across Conditions",
       x = "Total points (log)", y = "Message length (nchar)")





#### Message stats after processing ####

response_data = read.csv('../data/responses_processed.csv') %>% select(-X)
# response_data = response_data %>%
#   mutate(condition=ifelse(condition=='medium-1', 'medium', 
#                           ifelse(condition=='easy', 'high', 'low')))
# write.csv(response_data, file = '../data/responses_processed.csv')

ggplot(response_data, aes(x = condition, y = n_NAs, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 0.6) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
    position = position_nudge(x = 0.2)
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  theme_minimal() +
  # labs(title = "Numer of Patterns Reported per Condition",
  #      x = "Condition", y = "N Patterns") +
  theme(legend.position = "none")

anova_result <- aov(n_NAs ~ condition, data = response_data)
summary(anova_result)
shapiro.test(residuals(anova_result)) # passed
eta_squared(anova_result) # 0.11


ggplot(response_data, aes(x = condition, y = len_NAs, fill = condition)) +
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
  labs(title = "Number of Unknowns",
       x = "Compressibility", y = "N sentences") +
  theme(legend.position = "none", axis.text = element_text(size = 10))
anova_result <- aov(len_NAs ~ condition, data = response_data)
summary(anova_result)
eta_squared(anova_result) # 0.11



ggplot(response_data, aes(x = condition, y = len_rules, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 0.6) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
    position = position_nudge(x = 0.2)
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  theme_minimal() +
  # labs(title = "Numer of Patterns Reported per Condition",
  #      x = "Condition", y = "N Patterns") +
  theme(legend.position = "none")
anova_result <- aov(len_rules ~ condition, data = response_data)
summary(anova_result)


response_selected = response_data %>%
  select(id, condition, n_tips, len_tips, n_rules, len_rules, n_NAs, len_NAs, len_summary)

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

response_data_filtered %>%
  ggplot(aes(x = condition, y = len_rules, fill = condition)) +
  geom_half_violin(side = "l", width = 0.8, trim = FALSE, alpha = 0.6) +
  stat_summary(
    fun = mean, geom = "bar", width = 0.4, color = "black", fill = "white",
    position = position_nudge(x = 0.2)
  ) +
  stat_summary(
    fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
    position = position_nudge(x = 0.2)
  ) +
  geom_text(aes(label = id), vjust = -0.5, hjust = 0.5, size = 3)+
  theme_minimal() +
  # labs(title = "Numer of Patterns Reported per Condition",
  #      x = "Condition", y = "N Patterns") +
  theme(legend.position = "none")

anova_result <- aov(len_rules ~ condition, data = response_data_filtered)
summary(anova_result)
eta_squared(anova_result)
eta_squared_to_d <- function(eta_squared) {
  d <- sqrt(eta_squared) / sqrt(1 - eta_squared)
  return(d)
}
eta_squared_to_d(0.09)












