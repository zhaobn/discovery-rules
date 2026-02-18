library(dplyr)
library(tidyr)
options(scipen=999)

library(ggplot2)
library(patchwork)
library(ggdist)
library(ggpubr)
library(wesanderson)

action_data = read.csv('../data/action_data.csv') %>% 
  select(-X) %>%
  mutate(condition = factor(condition, levels = c('high', 'medium', 'low')))

shapes = c("triangle", "circle", "square", "diamond")
textures = c("plain", "checkered", "stripes", "dots")

action_data_simplified <- action_data %>%
  select(id, action_id, held, target, condition)

# Map object to feature indices
map_feature <- function(x) {
  parts <- strsplit(x, "_")[[1]]
  shape <- parts[1]
  texture <- parts[2]
    
  shape_idx <- if(is.na(shape)) {9} else {match(shape, shapes) - 1}       # From 1–4 to 0-3
  texture_idx <- if(is.na(texture)) {9} else {match(texture, textures) - 1} # From 1–4 to 0-3
    
  return(paste(c(shape_idx, texture_idx), collapse = ''))
}

action_data_simplified$held_coded <- sapply(action_data_simplified$held, map_feature)
action_data_simplified$target_coded <- sapply(action_data_simplified$target, map_feature)

action_data_simplified <- action_data_simplified %>%
  # collapse all extract to the same bucket
  mutate(held_coded = if_else(target_coded=='99', '99', held_coded)) %>%
  mutate(pair = paste0(held_coded, "-", target_coded))

# quick check - prop of extract over time
prop_extract <- action_data %>%
  mutate(is_extract = action == 'consume') %>%
  group_by(condition, action_id) %>%
  summarise(n_extract = sum(is_extract), n = n()) %>%
  mutate(extract_prop = n_extract/n)

ggplot(prop_extract, aes(x=action_id, y=extract_prop, color=condition)) +
  geom_line() +
  theme_bw()

fit <- aov(extract_prop ~ condition + factor(action_id), data = prop_extract)
summary(fit)


# number of unique pairs
combine_data <- action_data_simplified %>%
  filter(target_coded != '99') %>%
  group_by(condition, id) %>%
  mutate(combine_id = row_number()) %>%
  ungroup()

combine_unique_pairs <- combine_data %>%
  count(condition, action_id, pair) %>%
  count(condition, action_id)
  
plt_uniq_pairs_line <- ggplot(combine_unique_pairs, aes(x=action_id, y=n, color=condition)) +
  geom_line() +
  theme_bw()

fit <- aov(n ~ condition + factor(action_id), data = combine_unique_pairs)
summary(fit)

plt_uniq_pairs_box <- ggplot(combine_unique_pairs, aes(x = condition, y = n, color = condition)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  theme_bw()

plt_uniq_pairs_box + plt_uniq_pairs_line + 
  plot_layout(widths = c(1, 3)) &
  theme(legend.position = "bottom")


# How many unique pairs can an individual cover
combine_individual_coverage <- combine_data %>%
  count(condition, id, pair) %>%
  count(condition, id)

means <- combine_individual_coverage %>%
  group_by(condition) %>%
  summarise(mean_n = mean(n))

# Plot histogram with mean lines
ggplot(combine_individual_coverage, aes(x = n, fill = condition)) +
  geom_histogram(color = "white", binwidth = 1) +
  geom_vline(data = means, aes(xintercept = mean_n, color = condition),
             linetype = "dashed", size = 0.7) +
  theme_bw()

plt_ind_pairs_box <- ggplot(combine_individual_coverage, aes(x = condition, y = n, color = condition)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  theme_bw()

plt_uniq_pairs_box + plt_ind_pairs_box +
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom") 


fit <- aov(n ~ condition, data = combine_unique_pairs)
summary(fit) # 0.000000834 ***
fit <- aov(n ~ condition, data = combine_individual_coverage)
summary(fit) # 0.2


# (How much diversity is group over individuals?)

# (individual differences - repeat many times vs. cover many grounds)

# heat map
nums <- 0:3
objs <- expand.grid(nums, nums)
obj_list <- paste0(objs$Var1, "", objs$Var2)

pairs <- expand.grid(obj_list, obj_list)
pairs <- pairs[pairs$Var1 != pairs$Var2, ]
pair_list <- paste0(pairs$Var1, "-", pairs$Var2)

conditions <- action_data %>%
  count(condition) %>%
  pull(condition)

all_pair_data <- expand.grid(
  action_id = seq(40),
  pair = pair_list,
  condition = conditions
) 

all_ppt_pairs <- combine_data %>%
  select(id, condition, action_id, pair) %>%
  count(condition, action_id, pair)

all_pair_data_ppt <- all_pair_data %>%
  left_join(all_ppt_pairs, by=c('condition', 'action_id', 'pair')) %>%
  mutate(across(everything(), ~ replace_na(., 0)))

ggplot(all_pair_data_ppt, aes(x = action_id, y = pair, fill = n)) +
  geom_tile(color = "white") +
  facet_grid(~condition) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    x = "Action ID",
    y = "Pair",
    fill = "n"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 12),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )


pairs_ever_touched <- all_ppt_pairs %>%
  count(pair) %>%
  pull(pair) # 256 - all pairs



