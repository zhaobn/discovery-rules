library(dplyr)
library(tidyr)
options(scipen=999)

library(ggplot2)
library(ggdist)
library(ggpubr)
library(wesanderson)

action_data = read.csv('../data/action_data.csv') %>% 
  select(-X) %>%
  mutate(condition = factor(condition, levels = c('high', 'medium', 'low')))
combine_data = action_data %>% filter(action=='combine')


## Fusion proportion
combine_data %>% 
  group_by(id, condition) %>%
  summarise(n=n()) %>% 
  mutate(prop_combine=n/40) %>%
  group_by(condition) %>%
  summarise(
    sd_prop_combine = sd(prop_combine),
    avg_prop_combine = sum(prop_combine)/n(),
    n = n()
  )
prop_combine_data = combine_data %>% 
  group_by(id, condition) %>%
  summarise(n=n()) %>% 
  mutate(prop_combine=n/40)
summary(aov(prop_combine ~ condition, data = prop_combine_data)) #F(2,116)=0.12, p=.88.


## Repeated attempts
combine_checked <- combine_data %>%
  group_by(id, condition) %>%
  arrange(action_id, .by_group = TRUE) %>%
  mutate(
    held_same_as_prev = held == lag(held),
    target_same_as_prev = target == lag(target),
    repeated_both = held_same_as_prev & target_same_as_prev
  )
repetition_summary <- combine_checked %>%
  summarise(
    prop_repeated = mean(repeated_both, na.rm = TRUE),
    n_actions = n()
  )
ggplot(repetition_summary, aes(x = condition, y = prop_repeated, fill = condition)) +
  geom_boxplot() +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(y = "Prop. of Repetition", x = "Condition") +
  theme_minimal() +
  theme(legend.position = 'none')
repetition_summary %>%
  group_by(condition) %>%
  summarise(
    mean_prop = mean(prop_repeated),
    median_prop = median(prop_repeated),
    max_prop = max(prop_repeated),
    #min_prop_repeated = min(prop_repeated),
    #sd_prop_repeated = sd(prop_repeated),
    above_50 = sum(prop_repeated > 0.5),
    above_70 = sum(prop_repeated > 0.7),
    above_90 = sum(prop_repeated > 0.9),
    n_participants = n()
  )
ids_to_exclude <- repetition_summary %>%
  filter(prop_repeated > 0.5) %>%
  pull(id)
combine_cleaned <- combine_data %>%
  filter(!id %in% ids_to_exclude)


## Number of fusions in between successes
combine_data_indexed = combine_cleaned %>%
  group_by(id) %>%
  mutate(yield_index = cumsum(yield != "" & !is.na(yield))) %>%
  select(id, action_id, held, target, yield, yield_index, condition)

step_data = combine_data_indexed %>%
  group_by(condition, id, yield_index) %>%
  summarise(n = n()) %>%
  mutate(steps = ifelse(yield_index > 0, n-1, n)) 

step_data %>%
  group_by(id, condition) %>%
  summarise(step_ind = sum(steps)/n()) %>%
  group_by(condition) %>%
  summarise(
    sd_steps = sd(step_ind),
    mean_steps = sum(step_ind)/n(),
    n = n()
  )
step_ind = step_data %>%
  group_by(id, condition) %>%
  summarise(step_ind = sum(steps)/n()) 
summary(aov(step_ind ~ condition, data = step_ind)) #F(2,107)=18.71, p=1.08e-07

ggplot(step_ind, aes(x = condition, y = step_ind, fill = condition)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0, point_alpha = 0, justification = -0.3) +
  geom_boxplot(width = 0.2, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1) +
  labs(y = "Steps before success", x = "Condition") +
  theme_classic() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, size = 0.8),
    legend.position = 'none'
  ) +
  stat_compare_means(
    comparisons = list(
      c("high", "medium"),
      c("high", "low"),
      c("medium", "low")
    ),
    method = "t.test",
    label = "p.signif"
  )

## Number of fusions before first success
first_yield_data = combine_data_indexed%>% 
  group_by(condition, id) %>%
  summarise(before_first_yield = sum(yield_index == 0)) %>%
  arrange(id)
first_yield_data %>%
  group_by(condition) %>%
  summarise(
    sd_first_yield = sd(before_first_yield),
    mean_first_yield = sum(before_first_yield)/n(),
    n = n(),
  ) 
summary(aov(before_first_yield ~ condition, data = first_yield_data)) #F(2,107)=6.6, p=0.002

# first-go success
first_yield_data %>% 
  filter(before_first_yield==0) %>%
  count(condition) # 18:1:3

# not on first-go
first_more_data =  step_data %>% filter(yield_index<1)
summary(aov(n ~ condition, data = first_more_data)) #F(2,85)=0.6, p=.54
ggplot(first_more_data, aes(x = condition, y = n, fill = condition)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0, point_alpha = 0, justification = -0.3) +
  geom_boxplot(width = 0.2, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1) +
  labs(y = "Steps before 1st", x = "Condition") +
  theme_classic() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, size = 0.8),
    legend.position = 'none'
  )


# First success similarity
first_goer_ids = first_yield_data %>% filter(before_first_yield==0) %>% pull(id)
first_goer_choice = combine_data_indexed %>%
  filter(id %in% first_goer_ids) %>%
  filter(action_id==1)
all_first_choice = combine_data_indexed %>%
  filter(action_id==1)

# Function to split a column like 'circle_plain_0' into three parts
split_feature_string <- function(df, colname) {
  df %>%
    tidyr::separate(
      {{ colname }},
      into = paste0(colname, c("_shape", "_texture", "_level")),
      sep = "_",
      fill = "right",
      remove = FALSE
    )
}
all_first_choice <- all_first_choice %>%
  split_feature_string('held') %>%
  split_feature_string('target') %>%
  split_feature_string('yield') %>%
  mutate(across(everything(), ~ replace_na(.x, "")))


## Diversity in target objects
# distribution of shape combos
shape_combo_order <- c(
  "circle-circle", "diamond-diamond", "square-square", "triangle-triangle",
  "circle-diamond", "circle-square", "circle-triangle",
  "diamond-circle", "diamond-square", "diamond-triangle",
  "square-circle", "square-diamond", "square-triangle",
  "triangle-circle", "triangle-diamond", "triangle-square"
)
texture_combo_order <- c(
  "checkered-checkered", "dots-dots", "plain-plain", "stripes-stripes",
  "checkered-dots", "checkered-plain", "checkered-stripes",
  "dots-checkered", "dots-plain", "dots-stripes",
  "plain-checkered", "plain-dots", "plain-stripes",
  "stripes-checkered", "stripes-dots", "stripes-plain"
)

all_first_choice_extended = all_first_choice %>%
  mutate(shape_combo = paste0(held_shape, '-', target_shape),
         texture_combo = paste0(held_texture, '-', target_texture)) %>%
  mutate(shape_combo = factor(shape_combo, levels = shape_combo_order),
         texture_combo = factor(texture_combo, levels = texture_combo_order))

all_first_choice_extended %>% 
  count(condition, shape_combo) %>%
  group_by(condition) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = condition, y = prop, fill = shape_combo)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(
    values = c(
      # Vivid colors for same-shape combos
      "circle-circle" = "#2CA02C",
      "diamond-diamond" = "#8FBD5A",
      "square-square" = "#E8B219",
      "triangle-triangle" = "#F28E2D",
      
      # Grey-ish distinguishable hues for mixed-shape combos
      "circle-diamond" = "#b3b3b3",
      "circle-square" = "#969696",
      "circle-triangle" = "#bdbdbd",
      "diamond-circle" = "#cccccc",
      "diamond-square" = "#a6a6a6",
      "diamond-triangle" = "#d9d9d9",
      "square-circle" = "#b0b0b0",
      "square-diamond" = "#8c8c8c",
      "square-triangle" = "#e0e0e0",
      "triangle-circle" = "#c0c0c0",
      "triangle-diamond" = "#a8a8a8",
      "triangle-square" = "#d0d0d0"
    )
  ) +
  labs(y = "Proportion", x = "Condition", fill = "Shape Combo") +
  theme_minimal()

all_first_choice_extended %>% 
  count(condition, texture_combo) %>%
  group_by(condition) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = condition, y = prop, fill = texture_combo)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c(
    # Vivid colors for same-texture combos
    "checkered-checkered" = "#2CA02C",
    "dots-dots" = "#8FBD5A",
    "plain-plain" = "#E8B219",
    "stripes-stripes" = "#F28E2D",
    
    # Grey-ish distinguishable hues for mixed-texture combos
    "checkered-dots" = "#b3b3b3",
    "checkered-plain" = "#969696",
    "checkered-stripes" = "#bdbdbd",
    "dots-checkered" = "#cccccc",
    "dots-plain" = "#a6a6a6",
    "dots-stripes" = "#d9d9d9",
    "plain-checkered" = "#b0b0b0",
    "plain-dots" = "#8c8c8c",
    "plain-stripes" = "#e0e0e0",
    "stripes-checkered" = "#c0c0c0",
    "stripes-dots" = "#a8a8a8",
    "stripes-plain" = "#d0d0d0"
  )) +
  labs(y = "Proportion", x = "Condition", fill = "Texture Combo") +
  theme_minimal()


## Success similarity
all_success_combo = combine_cleaned %>%
  filter(nchar(yield) > 0) %>%
  split_feature_string('held') %>%
  split_feature_string('target') %>%
  split_feature_string('yield') %>%
  mutate(shape_combo = paste0(held_shape, '-', target_shape),
         texture_combo = paste0(held_texture, '-', target_texture)) %>%
  mutate(shape_combo = factor(shape_combo, levels = shape_combo_order),
         texture_combo = factor(texture_combo, levels = texture_combo_order))

all_success_combo %>% 
  count(condition, shape_combo) %>%
  group_by(condition) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = condition, y = prop, fill = shape_combo)) +
  geom_bar(stat = "identity") +
  labs(y = "Proportion", x = "Condition", fill = "Shape Combo") +
  theme_minimal()
all_success_combo %>% 
  count(condition, texture_combo) %>%
  group_by(condition) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = condition, y = prop, fill = texture_combo)) +
  geom_bar(stat = "identity") +
  labs(y = "Proportion", x = "Condition", fill = "Texture Combo") +
  theme_minimal()


## Holding the same object
# remove repeated lines
combine_cleaned_check <- combine_cleaned %>%
  group_by(id, condition) %>%
  arrange(action_id, .by_group = TRUE) %>%
  mutate(
    held_same_as_prev = held == lag(held),
    target_same_as_prev = target == lag(target),
    repeated_both = held_same_as_prev & target_same_as_prev
  )
combine_cleaned_removed <- combine_cleaned_check %>%
  filter(repeated_both == FALSE | is.na(repeated_both)) %>%
  select(id, action_id, held, target, yield, condition, held_same_as_prev, target_same_as_prev)
nrow(combine_cleaned_removed) / nrow(combine_cleaned_check) #0.9470071

held_indexed <- combine_cleaned_removed %>% group_by(id, condition) %>%
  mutate(
    held_change = held != lag(held, default = first(held)), # marks TRUE when held changes from the previous row.
    held_block = cumsum(held_change) #assigns a block index that increments each time held changes.
  ) %>%
  group_by(id, condition, held_block, held) %>%
  summarise(
    n_interaction = n_distinct(target),
    .groups = "drop"
  ) %>%
  group_by(id, condition) %>%
  mutate(held_id = row_number()) %>%
  select(-held_block) %>%
  ungroup()

ggplot(held_indexed, aes(x = n_interaction, fill = condition)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black", alpha = 0.7) +
  facet_wrap(~condition) +
  labs(
    x = "Number of unique targets per held item (consecutive blocks)",
    y = "Count",
    title = "Distribution of n_interaction per Condition"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

held_indexed %>%
  group_by(condition, id) %>%
  summarise(avg_interaction = sum(n_interaction)/n()) %>%
  ggplot(aes(x=condition, y=avg_interaction, fill=condition)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0, point_alpha = 0, justification = -0.3) +
  geom_boxplot(width = 0.2, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "black", color = "white") +
  theme_minimal() +
  theme(legend.position = 'none') +
  stat_compare_means(
    comparisons = list(
      c("high", "medium"),
      c("high", "low"),
      c("medium", "low")
    ),
    method = "t.test",
    label = "p.signif"
  )

held_indexed %>%
  group_by(condition, held_id) %>%
  summarise(avg_n_interaction = mean(n_interaction), .groups = "drop") %>%
  ggplot(aes(x = held_id, y = avg_n_interaction, color = condition)) +
  geom_point(size = 2) +
  geom_line(aes(group = condition), linewidth = 1) +
  labs(x = "Held Item Index", y = "Average Number of Interactions", color = "Condition") +
  theme_minimal() +
  theme(legend.position = 'top')



## Holding the previous discovery
combine_cleaned %>%
  select(id, action_id, condition, held, target, yield) %>% head()



# Ensure "" yields are treated as NA for filtering
reuse_yield <- combine_cleaned %>%
  filter(nchar(yield) > 0) %>%
  group_by(id) %>%
  mutate(yield_id = row_number()) %>%
  ungroup() %>%
  # For each row, get the next held for the same id
  mutate(
    next_held = lead(held),
    next_id = lead(id),
    next_held_matches_yield = (next_id == id) & (next_held == yield)
  ) %>%
  select(
    id, condition,
    held, target, yield, yield_id,
    next_held_matches_yield
  )

reuse_yield %>%
  group_by(id, condition) %>%
  summarise(yield_reuse_rate = sum(next_held_matches_yield)/n()) %>%
  ggplot(aes(x=condition, y=yield_reuse_rate, fill=condition)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0, point_alpha = 0, justification = -0.3) +
  geom_boxplot(width = 0.2, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "black", color = "white") +
  theme_minimal() +
  theme(legend.position = 'none') +
  stat_compare_means(
    comparisons = list(
      c("high", "medium"),
      c("high", "low"),
      c("medium", "low")
    ),
    method = "t.test",
    label = "p.signif"
  )


## Geo-distance
event_data = read.csv('../data/events_data.csv')

# quick check location - center is (7,7)
event_data %>% filter(event_id==1)

event_data_filtered = event_data %>%
  filter(substr(action, 0, 4) != 'move') %>%
  mutate(condition=ifelse(assignment=='easy', 'high', ifelse(assignment=='hard', 'low', 'medium'))) %>%
  mutate(condition = factor(condition, levels = c('high', 'medium', 'low')))

# distance to first picked-up item
first_pickups <- event_data_filtered %>% 
  filter(action=='pickUp') %>%
  select(id, event_id, action, x, y, condition) %>%
  group_by(id) %>%
  filter(event_id == min(event_id)) %>%
  filter(!id %in% ids_to_exclude) %>%
  ungroup() %>%
  mutate(mhd = abs(x-7) + abs(y-7))

ggplot(first_pickups, aes(x = mhd, fill = condition)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black", alpha = 0.7) +
  facet_wrap(~condition) +
  labs(
    x = "First pick-up distance",
    y = "Count",
    title = "Distribution of first pick-up distance per Condition"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

summary(aov(mhd ~ condition, data = first_pickups)) #F(2,107)=0.13, p=.88

# distance between combine actions
action_locations = event_data_filtered %>%
  filter(actionsLeft > -1) %>%
  filter(action %in% c('combine', 'consume')) %>%
  mutate(action_id = 40-actionsLeft) %>%
  select(id, action_id, action, x, y, currentlyCarrying, condition)
action_data_extended = action_data %>%
  left_join(action_locations, by=c('id', 'action_id', 'action', 'condition'))
#write.csv(action_data_extended, file = 'action_data_2.csv')

combine_geo = action_data_extended %>%
  filter(!id %in% ids_to_exclude & action=='combine') %>%
  select(id, action_id, action, x, y, held, target, yield, condition) %>%
  group_by(id) %>%
  arrange(action_id) %>%
  mutate(manhattan_dist = abs(x - lag(x)) + abs(y - lag(y))) %>%
  ungroup() %>%
  filter(!is.na(manhattan_dist))

ggplot(combine_geo, aes(x = manhattan_dist, fill = condition)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black", alpha = 0.7) +
  facet_wrap(~condition) +
  labs(
    x = "Combine move distance",
    y = "Count",
    title = "Distribution of combine move distance per Condition"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
summary(aov(manhattan_dist ~ condition, data = combine_geo)) #F(2,3529)=15.29, p=.0.000000244


combine_geo_ind = combine_geo %>%
  group_by(condition, id) %>%
  summarise(combine_mhd = sum(manhattan_dist)/n())
combine_geo_ind %>%
  ggplot(aes(x=condition, y=combine_mhd, fill=condition)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0, point_alpha = 0, justification = -0.3) +
  geom_boxplot(width = 0.2, outlier.shape = NA, color = "black") +
  geom_jitter(width = 0.1, alpha = 0.5, size = 1) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "black", color = "white") +
  theme_minimal() +
  theme(legend.position = 'none') +
  stat_compare_means(
    comparisons = list(
      c("high", "medium"),
      c("high", "low"),
      c("medium", "low")
    ),
    method = "t.test",
    label = "p.signif"
  )
summary(aov(combine_mhd ~ condition, data = combine_geo_ind)) #F(2,107)=4.139, p=.0186

combine_geo %>%
  group_by(id) %>%
  mutate(combine_id = row_number()) %>%
  group_by(condition, combine_id) %>%
  summarise(avg_dist = mean(manhattan_dist), .groups = "drop") %>%
  ggplot(aes(x = combine_id, y = avg_dist, color = condition)) +
  geom_point(size = 2) +
  geom_line(aes(group = condition), linewidth = 1) +
  labs(x = "Combine Action Index", y = "Average Travel Distance", color = "Condition") +
  theme_minimal() +
  theme(legend.position = 'bottom')








