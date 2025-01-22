library(dplyr)
library(tidyr)
library(ggplot2)
library(gghalves)

#### Pilot 3 ####
subject_data = read.csv('../data/pilot-3/subject_data.csv')
assignment_orders = c('easy', 'medium', 'hard')

subject_data = subject_data %>%
  mutate(condition=ifelse(condition=='medium-1', 'medium', condition)) %>%
  mutate(condition=factor(condition, levels=assignment_orders)) %>%
  select(id, condition, messageHow, messageRules, total_points, task_duration)

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


# Message length per condition
subject_data$message_length = nchar(subject_data$messageRules)
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


# Points and lengths
ggplot(subject_data, aes(x=total_points_log, y=message_length, color=condition)) +
  geom_point(aes(shape=condition), size=3) +
  geom_smooth(method = "lm", se = TRUE, aes(fill = condition), alpha = 0.1) +
  theme_minimal() +
  labs(title = "Total Points and Message Lengths across Conditions",
       x = "Total points (log)", y = "Message length (nchar)")


# Points per action



#### Pilots 1-2 ####
# load data
assignment_orders = c('easy-1', 'medium-1', 'hard-1',
                      'easy-2', 'medium-2', 'hard-2',
                      'medium-3', 'medium-4', 'hard-3')

subject_1 = read.csv('../data/pilot-1/subject_data.csv')
subject_1 = subject_1 %>% 
  mutate(assignment=ifelse(version==0.1, paste0(assignment, '-1'), paste0(assignment, '-2')))

subject_2 = read.csv('../data/pilot-2/subject_data.csv')
subject_2 = subject_2 %>% 
  mutate(assignment = case_when(
    assignment == 'medium-1' ~ 'medium-3',
    assignment == 'medium-2' ~ 'medium-4',
    assignment == 'hard' ~ 'hard-3',
    TRUE ~ assignment
  ))
subject_all = rbind(subject_1, subject_2)
subject_all$assignment = factor(subject_all$assignment, levels = assignment_orders)

subject_all$total_points_log = log(subject_all$total_points)

# all points
ggplot(subject_all, aes(x = assignment, y = total_points_log, fill = assignment)) +
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


# message lengths
subject_all$message_length = nchar(subject_all$message)
ggplot(subject_all, aes(x = assignment, y = message_length, fill = assignment)) +
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
  labs(title = "Message Nchar per Condition",
       x = "Condition", y = "Nchar") +
  theme(legend.position = "none")



# accumulated points over actions
action_1 = read.csv('../data/pilot-1/action_data.csv')
action_2 = read.csv('../data/pilot-2/action_data.csv')

action_1 = action_1 %>% 
  mutate(assignment=ifelse(version==0.1, paste0(assignment, '-1'), paste0(assignment, '-2')))
action_2 = action_2 %>% mutate(assignment = case_when(
  assignment == 'medium-1' ~ 'medium-3',
  assignment == 'medium-2' ~ 'medium-4',
  assignment == 'hard' ~ 'hard-3',
  TRUE ~ assignment
))

action_all = rbind(action_1, action_2)
action_all$assignment = factor(action_all$assignment, levels = assignment_orders)

action_all$action_id = as.numeric(substr(action_all$action_id, 5, nchar(action_all$action_id)))
action_all <- action_all %>%
  filter(
    !(version == 0.1 & action_id > 20) &
      !(version == 0.2 & action_id > 40) &
      !(version == 0.3 & action_id > 30)
  ) %>%
  group_by(id) %>% 
  arrange(action_id) %>%
  mutate(total_points = cumsum(points)) %>%
  mutate(total_points_log = ifelse(total_points<1, 0, 
                                   ifelse(total_points==1, 0.1, log(total_points)) ))

ggplot(action_all, aes(x=factor(action_id), y=total_points_log)) +
  geom_boxplot() + 
  facet_grid(assignment~.)

selected_actions = action_all %>%
  filter(assignment %in% c('easy-2', 'medium-3', 'hard-3')) %>%
  filter(action_id<31)
selected_actions_summary <- selected_actions %>%
  group_by(action_id, assignment) %>%
  summarise(
    mean_total_points_log = mean(total_points_log, na.rm = TRUE),
    se_total_points_log = sd(total_points_log, na.rm = TRUE) / sqrt(n())
  )
ggplot(selected_actions_summary, aes(x = action_id, y = mean_total_points_log, color = assignment)) +
  geom_line() +
  geom_ribbon(aes(ymin = mean_total_points_log - se_total_points_log, 
                  ymax = mean_total_points_log + se_total_points_log, 
                  fill = assignment), alpha = 0.2, color = NA) + 
  geom_point(aes(shape = assignment), size = 3) + 
  theme_minimal() +   
  labs(x = "Action ID", y = "Mean Log of Total Points") + 
  theme(legend.position = "bottom")   


# success rates over actions
selected_combinations <- selected_actions %>%
  filter(action == 'combine') %>%
  mutate(success = ifelse(nchar(yield) > 1, 1, 0)) %>%
  group_by(id) %>%
  mutate(
    combination_id = row_number(),
    cumulative_success = cumsum(success)
  ) %>%
  ungroup() %>%
  mutate(cumulative_success_rate = cumulative_success/combination_id)
combination_summary <- selected_combinations %>%
  group_by(combination_id, assignment) %>%
  summarise(
    mean_cumulative_success_rate = mean(cumulative_success_rate), 
    se_success_rate = sd(cumulative_success_rate, na.rm = TRUE) / sqrt(n()),  
    .groups = "drop"
  )
ggplot(combination_summary, aes(x = combination_id, y = mean_cumulative_success_rate, color = assignment)) +
  geom_line() +
  geom_point(aes(color = assignment), size = 3) +
  geom_ribbon(aes(ymin = mean_cumulative_success_rate - se_success_rate, 
                  ymax = mean_cumulative_success_rate + se_success_rate, 
                  fill = assignment), 
              alpha = 0.2, color = NA) +
  theme_minimal() +
  labs(x = "Combination ID", y = "Cumulative Success Rate") +
  theme(legend.position = "bottom")





