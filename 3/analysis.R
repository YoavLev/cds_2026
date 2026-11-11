# Step 2: table, plot and mixed model for the results from experiment.py
library(tidyverse)
library(lme4)

# 1. Read results and drop answers without a salary
raw <- read_csv("data/results.csv")
results <- raw |>
  filter(!is.na(salary_dkk), !is.na(hire_score)) |>
  mutate(origin = factor(origin, levels = c("Danish", "MENA", "Eastern European")))
cat("Dropped", nrow(raw) - nrow(results), "of", nrow(raw), "rows with no usable answer\n")

# 2. Table: mean salary and score by model, origin and job
summary_table <- results |>
  group_by(model, job, origin) |>
  summarise(mean_salary = mean(salary_dkk), sd_salary = sd(salary_dkk),
            mean_score = mean(hire_score), n = n(), .groups = "drop")
print(summary_table, n = Inf)
write_csv(summary_table, "summary_table.csv")

# 3. Plot: salary by origin, one panel per job (and one row per model)
ggplot(results, aes(x = origin, y = salary_dkk, fill = origin)) +
  geom_boxplot() +
  facet_grid(model ~ job) +
  labs(x = NULL, y = "Recommended monthly salary (DKK)") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("salary_boxplot.png", width = 10, height = 6)

# 4. Mixed models, fitted separately for each LLM
#    Main-effects model (H1, H2): the overall origin gap, averaged over jobs
#    Interaction model (H3): does the gap differ between jobs?
#    (in the interaction model, the origin rows are the gap for Nurse only)
for (m in unique(results$model)) {
  d <- filter(results, model == m)
  for (outcome in c("salary_dkk", "hire_score")) {
    cat("\n=====", m, "|", outcome, "=====\n")

    main <- lmer(reformulate(c("origin", "job", "(1 | name_id)"), outcome), data = d)
    est <- coef(summary(main))[c("originMENA", "originEastern European"), ]
    print(tibble(term = rownames(est),
                 estimate = est[, "Estimate"],
                 ci_low = est[, "Estimate"] - 1.96 * est[, "Std. Error"],
                 ci_high = est[, "Estimate"] + 1.96 * est[, "Std. Error"]))

    interaction <- lmer(reformulate(c("origin * job", "(1 | name_id)"), outcome), data = d)
    print(summary(interaction))
  }
}
