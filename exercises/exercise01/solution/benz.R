library(skimr)
library(DataExplorer) # Data manipulation
library(dplyr)
library(tidyverse)
library(magrittr) # piping
library(ggplot2) # Visualization
library(readr) # Read csv
library(stringr) # String manipulation

setwd("C:/Users/jlzmk/PycharmProjects/KI-Machine-Learning/Introduction to R")
options(scipen = 999)

# Read csv file----
benz_df <- read_csv("benzfinal.csv")

# Data cleaning----
benz_df$Manufacturer <- str_remove(benz_df$Manufacturer, "\\d+")
benz_df$Model <- gsub('[\\""]', '', benz_df$Model)

# Overview of Dataset ----
summary(benz_df)
skim(benz_df)

# Visualize missing data per feature ----
plot_missing(benz_df)

# Create new labels ----
benz_df$sports_car <- ifelse(benz_df$Ps > 300, 1, 0)

# Alternative
benz_df<- benz_df%>%
  mutate(sports_car= ifelse(benz_df$Ps > 300, 1, 0))

benz_df$luxury_car <- ifelse(benz_df$Ps > 400 & benz_df$Price > 200000, 1, 0)

# Filter for luxury sports cars ----
benz_df_sports_luxury <- subset(benz_df, luxury_car == 1 & sports_car == 1) #  is redundant
# Alternative
benz_df_sports_luxury <- benz_df %>% filter(luxury_car == 1)

# Filter for certain Mercedes Models ----
last_df <- subset(benz_df, Manufacturer == "Mercedes-Benz" & (Price > 300000 | Ps > 400),
                  select = c(Model, Km, Ps, Price))
# Alternative with piping
last_df <- benz_df %>% 
  filter(Manufacturer == "Mercedes-Benz" & (Price > 300000 | Ps > 400)) %>% 
  select(Model, Km, Ps, Price)

# PS per manufacturer ----
aggregate(Ps ~ Manufacturer, data = benz_df, FUN = mean)

# Calculate the number of cars per year ----
yearly_count <- benz_df %>%
  group_by(Year) %>%
  summarise(count = n())

# Visualize the results ----
ggplot(yearly_count, aes(x = Year, y = count)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "Number of Cars per Year",
    x = "Year",
    y = "Number of Cars"
  )


# Filter the models that begin with "GL" ----
model_gl <- benz_df %>%
  filter(str_detect(Model, "^GL"))
head(model_gl)

# Create new SUV column ----
benz_df <- benz_df %>%
  mutate(SUV = ifelse(str_detect(Model, "GL|X"), "yes", "no"))
head(benz_df)

# Plot ps histogramm ----
ggplot(benz_df, aes(x = Ps)) +
  geom_histogram(binwidth = 20, fill = "white", color = "black") +
  labs(title = "PS Distribution in Car Dataset",
       y = "PS",
       x = "Prevalence")

# Scatterplot price & ps ----
ggplot(benz_df, aes(x = Price, y = Ps, color = Manufacturer)) +
  geom_point(size = 3) +
  labs(title = "Scatterplot Price & Ps (per Manufacturer)",
       x = "Price",
       y = "Ps")

# Sort by Price descendant and show first ten rows ----
top_10_most_expensive <- benz_df %>%
  arrange(desc(Price)) %>%
  head(10)
top_10_most_expensive

# Group by Manufacturer and calculate max price ----
max_price_by_manufacturer <- benz_df %>%
  group_by(Manufacturer) %>%
  summarise(max_price = max(Price, na.rm = TRUE))
max_price_by_manufacturer

max_price_by_manufacturer <- benz_df %>% 
  group_by(Manufacturer) %>% 
  arrange(desc(Price)) %>% 
  slice(1)

# Welch Two Sample t-test, 95% Confidence Interval ----
t_test_result <- t.test(
  Price ~ Ps > 300,
  data = benz_df,
  var.equal = FALSE
)
t_test_result

# Calculate correlation between Ps and Price ----
cor(benz_df$Ps, benz_df$Price, use = "complete.obs")

# Filter for cars more expensive than 100000 ----
subset_data <- benz_df %>%
  filter(Price > 100000) %>%
  select(Manufacturer, Model, Price, Ps)

write.csv(subset_data, "subset_data.csv", row.names = FALSE)


# First Regression Model ----
lr_model_1 <- lm(Price ~ Ps + Manufacturer + Km + state + engine, benz_df)
summary(lr_model_1)

# Add/Remove Features ----
lr_model_2 <- lm(Price ~ Ps + Manufacturer + state + Year + Model, benz_df)
summary(lr_model_2)

# Filter outliers ----
benz_df <- benz_df[-c(215, 50, 224), ]

# Feature Engineering ----

# Split models in multiple classes
benz_df <- benz_df %>%
  mutate(Model_class = case_when(
    # SUVs
    Model %in% c("X2", "X3", "X4", "X5", "X6", "GLC", "GLE", "GLS", "G") ~ "SUV",
    # Sedans
    Model %in% c("1er", "2er", "3er", "4er", "5er", "6er", "7er", "C", "E", "S", "CLA", "CLS") ~ "Sedan",
    # Coupes
    Model %in% c("M2", "M3", "M4", "M5", "M6", "420d", "420i", "430", "430d", "430i", "440i", "640", "640d", "640i", "650i", "CL", "SL", "SLS", "SLR", "Z1", "Z3", "Z8") ~ "Coupe",
    # Sports cars
    Model %in% c("i8", "AMG", "M550i", "Brabus") ~ "Sports Car",
    # Vans
    Model %in% c("Sprinter", "SPRINTER", "V") ~ "Van",
    # Other categories
    .default = "Other"
  ))
benz_df <- benz_df %>%
  mutate(Ps_category = case_when(
    Ps < 150 ~ "Low",
    Ps >= 150 & Ps < 250 ~ "Medium",
    Ps >= 250 & Ps < 400 ~ "High",
    Ps >= 400 & Ps < 600 ~ "Performance",
    Ps >= 600 ~ "High Performance"
  ))
# -> doesn't lead to better performance :(

lr_model_3 <- lm(Price ~ Ps + Manufacturer + state + Year + Model, benz_df)
summary(lr_model_3)
plot(lr_model_3)

# Further Ideas
# Price Categories (classification vs Regression)
# Combine Features
# Step wise regression

