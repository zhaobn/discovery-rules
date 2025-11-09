
library(dplyr)
library(tidyverse)

library(ggplot2)
library(ggrepel)

library(purrr)
library(jsonlite)



# Re-arrange rules & export
response_data = read.csv('responses_processed.csv') %>% select(-X)


response_data_fmt = response_data %>%
  mutate(parsed = map(coded, ~ fromJSON(.x, flatten = TRUE))) %>%
  mutate(
    parsed = gsub('\n', '', parsed),
    parsed = gsub('  ', '', parsed)
  ) %>%
  mutate(parsed_list = map(parsed, ~ fromJSON(.x))) %>%
  mutate(
    tips = map_chr(parsed_list, possibly(~ .x[[1]][[1]], '')),
    patterns = map_chr(parsed_list, possibly(~ .x[[2]][[1]], '')),
    NA_col = map_chr(parsed_list, possibly(~ .x[[3]][[1]], '')),
    summary = map_chr(parsed_list, possibly(~ .x[[4]][[1]], ''))
  ) %>%
  select(-parsed, -parsed_list)


write.csv(response_data_fmt, 'responses_processed_2.csv')



# Top messages
message_data = read.csv('responses_processed_labeled.tsv', sep = '\t')

message_data_filtered = message_data %>% filter(!(isEnv == 1 & isRuleLike == 0))
nrow(message_data) - nrow(message_data_filtered) # 18


message_data_filtered = message_data_filtered %>%
  mutate(
    rawLenHow = nchar(messageHow),
    rawLenRule = nchar(messageRules)
  ) %>%
  mutate(
    lenTotal = rawLenHow + rawLenRule
  ) %>%
  mutate(
    lenHow = ifelse(
      isRuleLike == 1, 
        ifelse(isEnv == 1, rawLenRule, 0), 
        rawLenHow
    ),
    lenRule = ifelse(
      isRuleLike == 1, 
      ifelse(isEnv == 1, rawLenHow, rawLenRule + rawLenHow),
      rawLenRule
    )
  ) %>%
  mutate(total_points_log = log(total_points + 1))  


message_data_stats = message_data_filtered %>%
  select(id, condition, total_points, total_points_log, lenTotal, lenHow, lenRule, isRuleLike, isEnv, rawLenHow, rawLenRule)

message_data_long <- message_data_stats %>%
  select(id, condition, total_points, lenTotal, lenRule, lenHow) %>%
  pivot_longer(cols = c(lenTotal, lenRule, lenHow), 
               names_to = "len_type", 
               values_to = "nchar") %>%
  mutate(len_type = substr(len_type, 4, nchar(len_type)),
         condition = factor(condition, levels = c('easy', 'medium', 'hard')),
         len_type = factor(len_type, levels = c('Total', 'Rule', 'How'))) 

ggplot(message_data_long, aes(x=total_points, y=nchar, group=condition)) +
  geom_point(aes(color=condition, shape=condition), size=3) +
  geom_text_repel(aes(label=id), size=3) +
  theme_bw() +
  facet_grid(len_type~condition)










