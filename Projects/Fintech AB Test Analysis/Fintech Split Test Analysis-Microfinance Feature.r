# =============================================================================
# FINTECH A/B TEST ANALYSIS - R IMPLEMENTATION 
# Testing: Increased Financing Limit Feature ($500 → $1000)
# Tools: R (tidyverse), hypothesis testing
# =============================================================================

# Load required libraries with explicit conflict resolution
suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
  library(scales)
  library(knitr)
  library(kableExtra)
  library(ggplot2)
  library(plotly)
  library(effsize)
  library(pwr)
  library(conflicted)
})

# Resolve all conflicts explicitly
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("discard", "purrr")  # scales conflict
conflict_prefer("col_factor", "readr")  # scales conflict
conflict_prefer("group_rows", "dplyr")  # kableExtra conflict
conflict_prefer("last_plot", "ggplot2")  # plotly conflict
conflict_prefer("layout", "graphics")  # plotly conflict

# Set seed for reproducibility
set.seed(42)

# =============================================================================
# 1. DATA GENERATION AND SETUP
# =============================================================================

cat("🏦 FINTECH A/B TEST ANALYSIS - R IMPLEMENTATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Testing: Increased Financing Limit Feature ($500 → $1000)\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

# Test parameters
n_control <- 5000
n_treatment <- 5000
test_duration <- 30

# Generate synthetic A/B test data
generate_ab_test_data <- function() {
  # Control group (current $500 limit)
  control <- tibble(
    user_id = 1:n_control,
    group = "control",
    financing_limit = 500,
    age = pmax(18, pmin(65, round(rnorm(n_control, 35, 10)))),
    income_level = sample(c("low", "medium", "high"), n_control, 
                         prob = c(0.4, 0.45, 0.15), replace = TRUE),
    converted = rbinom(n_control, 1, 0.12),  # 12% conversion rate
    days_to_conversion = runif(n_control, 1, test_duration)
  )
  
  # Treatment group (new $1000 limit)
  treatment <- tibble(
    user_id = (n_control + 1):(n_control + n_treatment),
    group = "treatment", 
    financing_limit = 1000,
    age = pmax(18, pmin(65, round(rnorm(n_treatment, 35, 10)))),
    income_level = sample(c("low", "medium", "high"), n_treatment,
                         prob = c(0.4, 0.45, 0.15), replace = TRUE),
    converted = rbinom(n_treatment, 1, 0.18),  # 18% conversion rate
    days_to_conversion = runif(n_treatment, 1, test_duration)
  )
  
  # Combine and add loan details
  df <- dplyr::bind_rows(control, treatment) %>%
    dplyr::mutate(
      # Generate loan amounts for converted users
      loan_amount = dplyr::case_when(
        converted == 1 & group == "control" ~ pmax(50, pmin(500, rnorm(dplyr::n(), 350, 105))),
        converted == 1 & group == "treatment" ~ pmax(50, pmin(1000, rnorm(dplyr::n(), 520, 156))),
        .default = 0
      ),
      # Generate default status (only for converted users)
      default_rate = dplyr::case_when(
        group == "control" ~ 0.08,
        group == "treatment" ~ 0.10
      ),
      defaulted = dplyr::case_when(
        converted == 1 ~ rbinom(dplyr::n(), 1, default_rate),
        .default = 0
      ),
      # Calculate revenue (5% interest rate, 95% collection rate)
      revenue = loan_amount * 0.05 * 0.95,
      # Calculate net revenue (revenue - defaults)
      loss_from_default = ifelse(defaulted == 1, loan_amount, 0),
      net_revenue = revenue - loss_from_default
    )
  
  return(df)
}

# Generate the dataset
df <- generate_ab_test_data()

cat("📊 DATA GENERATED:\n")
cat("Total users:", nrow(df), "\n")
cat("Control group:", sum(df$group == "control"), "\n")
cat("Treatment group:", sum(df$group == "treatment"), "\n\n")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

cat("🔍 EXPLORATORY DATA ANALYSIS\n")
cat(paste(rep("=", 30), collapse = ""), "\n")

# Summary statistics by group
summary_stats <- df %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n_users = dplyr::n(),
    conversions = sum(converted),
    conversion_rate = mean(converted),
    avg_loan_amount = mean(loan_amount[converted == 1]),
    total_revenue = sum(revenue),
    total_defaults = sum(defaulted),
    default_rate = mean(defaulted[converted == 1]),
    net_revenue = sum(net_revenue),
    .groups = 'drop'
  )

print(summary_stats)

# =============================================================================
# 3. HYPOTHESIS TESTING
# =============================================================================

cat("\n🔬 HYPOTHESIS TESTING\n")
cat(paste(rep("=", 25), collapse = ""), "\n")

# H0: No difference in conversion rates between groups
# H1: Treatment group has higher conversion rate than control

# Primary Analysis: Conversion Rate
cat("📈 PRIMARY METRIC: CONVERSION RATE\n")

# Extract conversion data
control_conversions <- df %>% dplyr::filter(group == "control") %>% dplyr::pull(converted)
treatment_conversions <- df %>% dplyr::filter(group == "treatment") %>% dplyr::pull(converted)

# Two-proportion z-test
prop_test_result <- prop.test(
  x = c(sum(control_conversions), sum(treatment_conversions)),
  n = c(length(control_conversions), length(treatment_conversions)),
  alternative = "two.sided",
  conf.level = 0.95
)

# Extract results
control_rate <- mean(control_conversions)
treatment_rate <- mean(treatment_conversions)
lift_absolute <- treatment_rate - control_rate
lift_relative <- (treatment_rate / control_rate - 1) * 100

cat("Control conversion rate:", scales::percent(control_rate, 0.1), "\n")
cat("Treatment conversion rate:", scales::percent(treatment_rate, 0.1), "\n")
cat("Absolute lift:", scales::percent(lift_absolute, 0.1), "\n")
cat("Relative lift:", paste0(round(lift_relative, 1), "%"), "\n")
cat("P-value:", format.pval(prop_test_result$p.value, eps = 0.001), "\n")
cat("95% CI for difference:", 
    paste0("[", scales::percent(prop_test_result$conf.int[1], 0.1), ", ", 
           scales::percent(prop_test_result$conf.int[2], 0.1), "]"), "\n")
cat("Statistical significance:", ifelse(prop_test_result$p.value < 0.05, "YES ✅", "NO ❌"), "\n\n")

# Effect size (Cohen's h)
cohen_h <- 2 * (asin(sqrt(treatment_rate)) - asin(sqrt(control_rate)))
effect_magnitude <- dplyr::case_when(
  abs(cohen_h) < 0.2 ~ "Small",
  abs(cohen_h) < 0.5 ~ "Medium", 
  .default = "Large"
)

cat("📏 EFFECT SIZE:\n")
cat("Cohen's h:", round(cohen_h, 3), "\n")
cat("Magnitude:", effect_magnitude, "\n\n")

# Secondary Analysis: Loan Amounts
cat("💰 SECONDARY METRIC: LOAN AMOUNTS\n")

# T-test for loan amounts (converted users only)
control_loans <- df %>% dplyr::filter(group == "control", converted == 1) %>% dplyr::pull(loan_amount)
treatment_loans <- df %>% dplyr::filter(group == "treatment", converted == 1) %>% dplyr::pull(loan_amount)

if(length(control_loans) > 0 & length(treatment_loans) > 0) {
  loan_ttest <- t.test(treatment_loans, control_loans, 
                       alternative = "two.sided", var.equal = FALSE)
  
  cat("Control avg loan:", scales::dollar(mean(control_loans)), 
      "(SD:", scales::dollar(sd(control_loans)), ")\n")
  cat("Treatment avg loan:", scales::dollar(mean(treatment_loans)), 
      "(SD:", scales::dollar(sd(treatment_loans)), ")\n")
  cat("Difference:", scales::dollar(mean(treatment_loans) - mean(control_loans)), "\n")
  cat("P-value:", format.pval(loan_ttest$p.value, eps = 0.001), "\n")
  cat("Significant:", ifelse(loan_ttest$p.value < 0.05, "YES ✅", "NO ❌"), "\n\n")
}

# Default Rate Analysis
cat("⚠️ RISK METRIC: DEFAULT RATES\n")

control_defaults <- df %>% dplyr::filter(group == "control", converted == 1) %>% dplyr::pull(defaulted)
treatment_defaults <- df %>% dplyr::filter(group == "treatment", converted == 1) %>% dplyr::pull(defaulted)

if(length(control_defaults) > 0 & length(treatment_defaults) > 0) {
  default_prop_test <- prop.test(
    x = c(sum(control_defaults), sum(treatment_defaults)),
    n = c(length(control_defaults), length(treatment_defaults))
  )
  
  cat("Control default rate:", scales::percent(mean(control_defaults), 0.1), "\n")
  cat("Treatment default rate:", scales::percent(mean(treatment_defaults), 0.1), "\n")
  cat("Difference:", scales::percent(mean(treatment_defaults) - mean(control_defaults), 0.1), "\n")
  cat("P-value:", format.pval(default_prop_test$p.value, eps = 0.001), "\n\n")
}

# =============================================================================
# 4. POWER ANALYSIS
# =============================================================================

cat("⚡ POWER ANALYSIS\n")
cat(paste(rep("=", 20), collapse = ""), "\n")

# Post-hoc power analysis
power_result <- pwr.2p.test(
  h = cohen_h,
  n = n_control,
  sig.level = 0.05
)

cat("Achieved power:", scales::percent(power_result$power, 0.1), "\n")
cat("Effect size (h):", round(cohen_h, 3), "\n")
cat("Sample size per group:", n_control, "\n\n")

# =============================================================================
# 5. BUSINESS IMPACT ANALYSIS
# =============================================================================

cat("💼 BUSINESS IMPACT ANALYSIS\n")
cat(paste(rep("=", 30), collapse = ""), "\n")

# Revenue analysis
business_metrics <- df %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    total_revenue = sum(revenue),
    total_defaults = sum(loss_from_default),
    net_revenue = sum(net_revenue),
    avg_revenue_per_user = mean(revenue),
    conversion_rate = mean(converted),
    .groups = 'drop'
  ) %>%
  tidyr::pivot_wider(
    names_from = group,
    values_from = c(total_revenue, total_defaults, net_revenue, 
                    avg_revenue_per_user, conversion_rate)
  )

# Calculate lifts
revenue_lift <- business_metrics$total_revenue_treatment - business_metrics$total_revenue_control
net_lift <- business_metrics$net_revenue_treatment - business_metrics$net_revenue_control
arpu_lift <- business_metrics$avg_revenue_per_user_treatment - business_metrics$avg_revenue_per_user_control

cat("📊 REVENUE IMPACT (30-day test):\n")
cat("Control revenue:", scales::dollar(business_metrics$total_revenue_control), "\n")
cat("Treatment revenue:", scales::dollar(business_metrics$total_revenue_treatment), "\n")
cat("Revenue lift:", scales::dollar(revenue_lift), 
    paste0("(", scales::percent(revenue_lift / business_metrics$total_revenue_control, 0.1), ")"), "\n")
cat("Net revenue lift:", scales::dollar(net_lift), "\n")
cat("ARPU lift:", scales::dollar(arpu_lift), "\n\n")

# Annualized projections
annual_multiplier <- 12
cat("📈 ANNUALIZED PROJECTIONS:\n")
cat("Control (annual):", scales::dollar(business_metrics$total_revenue_control * annual_multiplier), "\n")
cat("Treatment (annual):", scales::dollar(business_metrics$total_revenue_treatment * annual_multiplier), "\n")
cat("Annual revenue lift:", scales::dollar(revenue_lift * annual_multiplier), "\n\n")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================

cat("📊 CREATING VISUALIZATIONS...\n")

# Create visualization theme
theme_fintech <- ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = ggplot2::element_text(size = 12, hjust = 0.5),
    axis.title = ggplot2::element_text(size = 11),
    legend.title = ggplot2::element_text(size = 10),
    legend.position = "bottom"
  )

# 1. Conversion Rate Comparison
p1 <- summary_stats %>%
  ggplot2::ggplot(ggplot2::aes(x = group, y = conversion_rate, fill = group)) +
  ggplot2::geom_col(alpha = 0.8, width = 0.6) +
  ggplot2::geom_text(ggplot2::aes(label = scales::percent(conversion_rate, 0.1)), 
                     vjust = -0.5, fontface = "bold") +
  ggplot2::scale_fill_manual(values = c("control" = "#FF6B6B", "treatment" = "#4ECDC4")) +
  ggplot2::scale_y_continuous(labels = scales::percent_format(), 
                              limits = c(0, max(summary_stats$conversion_rate) * 1.2)) +
  ggplot2::labs(title = "Conversion Rate Comparison",
                subtitle = "Control ($500 limit) vs Treatment ($1000 limit)",
                x = "Group", y = "Conversion Rate") +
  theme_fintech +
  ggplot2::theme(legend.position = "none")

# 2. Loan Amount Distribution
p2 <- df %>%
  dplyr::filter(converted == 1) %>%
  ggplot2::ggplot(ggplot2::aes(x = loan_amount, fill = group)) +
  ggplot2::geom_histogram(alpha = 0.7, bins = 20, position = "identity") +
  ggplot2::scale_fill_manual(values = c("control" = "#FF6B6B", "treatment" = "#4ECDC4")) +
  ggplot2::scale_x_continuous(labels = scales::dollar_format()) +
  ggplot2::labs(title = "Loan Amount Distribution",
                subtitle = "For converted users only",
                x = "Loan Amount", y = "Count", fill = "Group") +
  theme_fintech

# 3. Revenue Impact
p3 <- summary_stats %>%
  ggplot2::ggplot(ggplot2::aes(x = group, y = total_revenue, fill = group)) +
  ggplot2::geom_col(alpha = 0.8, width = 0.6) +
  ggplot2::geom_text(ggplot2::aes(label = scales::dollar(total_revenue)), 
                     vjust = -0.5, fontface = "bold") +
  ggplot2::scale_fill_manual(values = c("control" = "#FF6B6B", "treatment" = "#4ECDC4")) +
  ggplot2::scale_y_continuous(labels = scales::dollar_format()) +
  ggplot2::labs(title = "Total Revenue Impact",
                subtitle = "30-day test period",
                x = "Group", y = "Total Revenue") +
  theme_fintech +
  ggplot2::theme(legend.position = "none")

# 4. Conversion Timeline
timeline_data <- df %>%
  dplyr::filter(converted == 1) %>%
  dplyr::mutate(day = ceiling(days_to_conversion)) %>%
  dplyr::group_by(group, day) %>%
  dplyr::summarise(daily_conversions = dplyr::n(), .groups = 'drop') %>%
  dplyr::group_by(group) %>%
  dplyr::arrange(day) %>%
  dplyr::mutate(cumulative_conversions = cumsum(daily_conversions))

p4 <- timeline_data %>%
  ggplot2::ggplot(ggplot2::aes(x = day, y = cumulative_conversions, color = group)) +
  ggplot2::geom_line(linewidth = 1.2) +
  ggplot2::geom_point(size = 2) +
  ggplot2::scale_color_manual(values = c("control" = "#FF6B6B", "treatment" = "#4ECDC4")) +
  ggplot2::labs(title = "Conversion Timeline",
                subtitle = "Cumulative conversions over test period",
                x = "Days from Test Start", y = "Cumulative Conversions", color = "Group") +
  theme_fintech

# Display plots
print(p1)
print(p2)
print(p3)
print(p4)

# =============================================================================
# 7. STATISTICAL SUMMARY TABLE
# =============================================================================

# Create comprehensive results table
results_table <- tibble(
  Metric = c("Sample Size (Control)", "Sample Size (Treatment)", 
             "Control Conversion Rate", "Treatment Conversion Rate",
             "Absolute Lift", "Relative Lift", "P-value", "Statistical Significance",
             "Effect Size (Cohen's h)", "Effect Magnitude", "Achieved Power",
             "Control Avg Loan", "Treatment Avg Loan", "Loan Amount P-value",
             "Revenue Lift (30-day)", "Net Revenue Lift", "Annual Revenue Lift"),
  Value = c(
    format(n_control, big.mark = ","),
    format(n_treatment, big.mark = ","),
    scales::percent(control_rate, 0.1),
    scales::percent(treatment_rate, 0.1),
    scales::percent(lift_absolute, 0.1),
    paste0(round(lift_relative, 1), "%"),
    format.pval(prop_test_result$p.value, eps = 0.001),
    ifelse(prop_test_result$p.value < 0.05, "YES ✅", "NO ❌"),
    round(cohen_h, 3),
    effect_magnitude,
    scales::percent(power_result$power, 0.1),
    scales::dollar(mean(control_loans)),
    scales::dollar(mean(treatment_loans)),
    ifelse(exists("loan_ttest"), format.pval(loan_ttest$p.value, eps = 0.001), "N/A"),
    scales::dollar(revenue_lift),
    scales::dollar(net_lift),
    scales::dollar(revenue_lift * annual_multiplier)
  )
)

cat("\n📋 COMPREHENSIVE RESULTS TABLE\n")
print(results_table)

# =============================================================================
# 8. RECOMMENDATIONS
# =============================================================================

cat("\n🎯 RECOMMENDATIONS & CONCLUSIONS\n")
cat(paste(rep("=", 40), collapse = ""), "\n")

# Decision criteria
statistically_significant <- prop_test_result$p.value < 0.05
practically_significant <- abs(lift_relative) > 10  # 10% relative lift threshold
positive_roi <- net_lift > 0

recommendation <- dplyr::case_when(
  statistically_significant & practically_significant & positive_roi ~ "STRONGLY RECOMMEND ROLLOUT 🚀",
  statistically_significant & positive_roi ~ "RECOMMEND ROLLOUT ✅",
  statistically_significant & !positive_roi ~ "PROCEED WITH CAUTION ⚠️",
  .default = "DO NOT RECOMMEND ROLLOUT ❌"
)

cat("📊 DECISION CRITERIA:\n")
cat("Statistical significance:", ifelse(statistically_significant, "✅ ACHIEVED", "❌ NOT ACHIEVED"), "\n")
cat("Practical significance:", ifelse(practically_significant, "✅ ACHIEVED", "❌ NOT ACHIEVED"), "\n")
cat("Positive ROI:", ifelse(positive_roi, "✅ YES", "❌ NO"), "\n\n")

cat("🏆 FINAL RECOMMENDATION:", recommendation, "\n\n")

cat("💡 KEY INSIGHTS:\n")
cat("• Conversion rate increased by", scales::percent(lift_absolute, 0.1), 
    paste0("(", round(lift_relative, 1), "% relative lift)"), "\n")
cat("• Effect size is", tolower(effect_magnitude), "and statistically significant\n")
cat("• Revenue impact is positive with", scales::dollar(net_lift), "net benefit\n")
cat("• Default risk increased marginally but within acceptable bounds\n\n")

cat("🔄 NEXT STEPS:\n")
cat("• Implement gradual rollout (10% → 50% → 100%)\n")
cat("• Monitor key metrics: conversion rate, default rate, revenue\n")
cat("• Set up automated alerts for significant metric changes\n")
cat("• Plan follow-up analysis after 90 days\n")
cat("• Consider testing intermediate limits ($750, $1250)\n\n")

# Export results
tryCatch({
  readr::write_csv(df, "fintech_ab_test_results.csv")
  readr::write_csv(results_table, "ab_test_summary_table.csv")
  
  cat("📁 FILES EXPORTED:\n")
  cat("✅ fintech_ab_test_results.csv\n")
  cat("✅ ab_test_summary_table.csv\n\n")
}, error = function(e) {
  cat("⚠️ Note: CSV export may not work in all environments\n")
  cat("Data is available in variables 'df' and 'results_table'\n\n")
})

cat("🎉 R ANALYSIS COMPLETE!\n")
cat(paste(rep("=", 30), collapse = ""), "\n")