library(dplyr)
library(effectsize)

subject_data = read.csv('../data/pilot-3/subject_data.csv')

# total points
anova_result <- aov(total_points ~ assignment, data = subject_data)
summary(anova_result)
shapiro.test(residuals(anova_result)) # violated

subject_data$total_points_log <- log(subject_data$total_points + 1)
anova_result <- aov(total_points_log ~ assignment, data = subject_data)
summary(anova_result)
shapiro.test(residuals(anova_result)) # passed, W=0.96, p=0.2

eta_squared(anova_result) # 0.16

# message lengths
subject_data$rule_length = nchar(subject_data$messageRules)
anova_result <- aov(rule_length ~ assignment, data = subject_data)
summary(anova_result)
shapiro.test(residuals(anova_result)) # violated

kruskal.test(rule_length ~ assignment, data = subject_data)
chi_sq <- 0.46118
n <- nrow(subject_data)
eta_squared_kruskal <- chi_sq / (n - 1)
eta_squared_kruskal

