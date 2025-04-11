

#####*************************Time Series Models***************************#####
#
# This lab contains 2 major parts:
#   *A- Univariate: 
#       We learn how to model TS in a univariate manner.
#       learn how to make decision of best model Vs. aggregation --> depending on
#       the nature of the data.
#   
#   *B- Multivariate:
#       Not all models can deal with Mutlivariate Datasets
#       Add Promotion as new Variable and re-evaluate the model
#####**********************************************************************#####


# A- Univariate----

# 1 Bibliotheken----
# Date and time manipulation
library(lubridate)

# Tools for working with time series data
library(timetk)

# Data manipulation and visualization
library(tidyverse)

# Framework for time series forecasting
library(modeltime)

# Unified interface to many models
library(parsnip)

# High-quality vector graphics device
library(Cairo)

# Simplifying complex code
library(magrittr)

# Create animations with ggplot2
library(gganimate)

# Graphics devices toolkit
library(gdtools)

# Benchmark datasets for machine learning
library(mlbench)

# Tools for modeling and machine learning
library(tidymodels)

# Data visualization and manipulation (again)
library(tidyverse)

# Kernel-based machine learning algorithms
library(kernlab)

# Unified interface for modeling and machine learning
library(workflows)

# Hyperparameter tuning
library(tune)

# Visualization of correlation matrices
library(corrplot)

# 2 Datensatz----
Party_df <- read_csv("TS_BSP_PARTY_store9/Party.csv")

Party_df %>% glimpse()
# * TS-Vizualisation----
Party_df %>%
  mutate(date=as.Date(date)) %>%   # be sure that data column is formatted as a date
  plot_time_series(date, sales,    # date column, target value
                   .interactive=TRUE,
                   .title="Store_9_transactions_ Celebrity_Articles",
                   .smooth = TRUE)



# Remarks----
# You can also aggregate the data into weekly / monthly / yearly depending on the problemset, Target and data quntity
# Exp:
Weekly_tbl<- Party_df %>% 
  summarise_by_time(date,.by = "week",
                    Weekly_sales= sum(sales),
                    weekly_promo= sum(onpromotion)) 

Weekly_tbl %>% 
  plot_time_series(date,Weekly_sales,    # date column, target value
                   .interactive=TRUE,
                   .title="Weekly_sales_Celebration",
                   .smooth = TRUE)


# Discuss the right solution: Imputation with NA conversion / remove specific time window
# 2 data cleaning, imputation and conversion to ts format----
Party_tbl<- Party_df %>%
  #  replace all 0 with NA --> Is that correct?
  mutate(sales=ifelse(sales==0,NA,sales))%>%
  mutate(sales=ts_impute_vec(sales,period = 12))

plot_time_series(Party_tbl, date,sales,
                 .interactive = TRUE,
                 .title = " Celebration articles with imputation")



# Löschen des Zeitraums Jan 13 bis Jan 14
clean_party <- Party_tbl[Party_tbl$date >= "2014-01-01", ]


Party_tbl_clean<- clean_party %>%
  #  remove 2013- Jan_2014
  mutate(sales=ifelse(sales==0,NA,sales))%>%
  mutate(sales=ts_impute_vec(sales,period = 12))

plot_time_series(Party_tbl_clean, date,sales,
                 .interactive = TRUE,
                 .title = " Celebration articles with imputation Starting Januar 2014")

#  3 Modeling----
#  * split data for ML

splits<- time_series_split( Party_tbl_clean, assess = "1 year", cumulative = TRUE) # 1 year for testing

splits %>%
  tk_time_series_cv_plan() %>%   # fct top interface TS
  group_by(date) %>%
  plot_time_series_cv_plan(date, sales,
                           .interactive = TRUE,
                           .title = "Celebration articles- Cross validation - ab 1-2014")

# * ARIMA

arima_model<- arima_reg()%>%
  set_engine("auto_arima")%>%
  fit(
    sales~date,
    data= training(splits)
  )

# * Linear Regression

LM_model<- linear_reg()%>%
  set_engine("lm")%>%
  fit(
    sales ~ as.numeric(date) + month(date, label = TRUE), #as numeric for trends
    data=training(splits)
  )

# * Linear Regression without trend
Lm_model_no_trends<- linear_reg()%>%
  set_engine("lm")%>%
  fit(
    sales~ month(date, label = TRUE),
    data=training(splits)
  )


# * Prophet
Prophet_model<- prophet_reg() %>%
  set_engine("prophet")%>%
  fit(
    sales~ date,
    data=training(splits)
  )

# * Random Forrest

RF_model<- rand_forest(mode = "regression") %>%
  set_engine("randomForest") %>%
  fit(
    sales~ as.numeric(date)+ month(date, label = TRUE),
    data=training(splits)
  )

# * XGBoost

XgBoost_model<-boost_tree(mode = "regression") %>%
  set_engine("xgboost") %>%
  fit(
    sales~ as.numeric(date)+ month(date, label = TRUE),
    data=training(splits)
  )


# * Support vector machine SVM_ polynomial

SVM_poly_model<- svm_poly(mode = "regression")%>%
  set_engine("kernlab")%>%
  fit(
    sales~ as.numeric(date)+ month(date, label = TRUE),
    data= training(splits)
  )


# SVM RBF - fitting model via spark

SVM_rbf_model<- svm_rbf(mode = "regression")%>%
  set_engine("kernlab")%>%
  fit(
    sales~ as.numeric(date)+ month(date, label = TRUE),
    data= training(splits)
  )

#******************************************************** Anmerkung:****************************************
# Unterschied zwischen SVM mit Polynomial Kernel (SVM_poly) und SVM mit RBF Kernel (SVM_rbf)
# 
# 1. **SVM mit Polynomial Kernel (SVM_poly)**:
#    - **Kernel-Typ**: Polynomialer Kernel (polynomiale Funktion)
#    - **Funktionsweise**: Der polynomiale Kernel transformiert die Daten in eine höhere Dimension und modelliert eine polynomiale Entscheidungsgrenze (z.B. quadratisch, kubisch).
#    - **Mathematisch**: K(x, y) = (x · y + c)^d, wobei d der Grad des Polynoms und c eine Konstante ist.
#    - **Wann verwenden?**:
#        - Wenn die Beziehung zwischen den Eingabedaten und der Zielvariable eine **polynomiale Struktur** aufweist (z.B. quadratische oder kubische Beziehung).
#        - Gut geeignet für **interaktive Effekte** zwischen den Merkmalen.
#        - Wenn keine komplexen nicht-linearen Beziehungen vermutet werden.
#        - Beispiel: Energieverbrauch in Abhängigkeit von Temperatur, bei dem die Beziehung quadratisch sein könnte.
#
# 2. **SVM mit RBF Kernel (SVM_rbf)**:
#    - **Kernel-Typ**: Radial Basis Function (RBF) Kernel (Exponentialfunktion)
#    - **Funktionsweise**: Der RBF-Kernel projiziert die Daten in einen höherdimensionalen Raum und misst den Abstand zwischen den Punkten, um eine **radiale Entscheidungsgrenze** zu erstellen.
#    - **Mathematisch**: K(x, y) = exp(-||x - y||^2 / (2σ^2)), wobei σ den Einflussbereich der Datenpunkte steuert.
#    - **Wann verwenden?**:
#        - Wenn die Beziehung zwischen den Eingabedaten und der Zielvariable **nicht-linear** und **komplex** ist.
#        - Wenn keine genaue Vorstellung von der Form der Beziehung besteht und das Modell flexibel genug sein muss, um komplexe Muster zu erkennen.
#        - Besonders geeignet für **hochdimensionale** Daten und Probleme mit komplexen, nicht-linearen Zusammenhängen.
#        - Beispiel: Klassifikation von Bilddaten, bei denen komplexe Muster durch den RBF-Kernel besser abgebildet werden.
#
# 3. **Wichtige Unterschiede**:
#    | **Aspekt**                           | **SVM mit Polynomial Kernel (SVM_poly)**                         | **SVM mit RBF Kernel (SVM_rbf)**                                                       |
#    |--------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------------------|
#    | **Kernel-Typ**                       | Polynomial Kernel (polynomiale Funktion)                         | Radial Basis Function (RBF) Kernel (Exponentialfunktion)                               |
#    | **Beziehung**                        | Modelliert polynomiale Beziehungen (z.B. quadratisch, kubisch)   | Modelliert nicht-lineare, radiale Beziehungen                                          |
#    | **Komplexität**                      | Weniger flexibel, da nur polynomiale Transformationen            | Sehr flexibel und kann sehr komplexe, nicht-lineare Muster modellieren                 |
#    | **Verwendung**                       | Wenn die Daten eine **polynomiale Beziehung** aufweisen          | Wenn die Daten **nicht-lineare und komplexe Muster** aufweisen                         |
#    | **Verhalten bei großen Dimensionen** | Kann bei vielen Merkmalen schwieriger zu optimieren sein         | Sehr gut geeignet für **hochdimensionale** Daten                                       |
#    | **Interpretierbarkeit**              | Die resultierende Entscheidungsgrenze ist ein Polynom            | Die Entscheidungsgrenze basiert auf dem Abstand zu Punkten, schwerer zu interpretieren |
#
# 4. **Wann verwendet man welches Modell?**:
#    - **Verwenden Sie den Polynomial Kernel (SVM_poly)**:
#        - Wenn Sie eine **polynomiale Beziehung** zwischen den Merkmalen und der Zielgröße erwarten (z.B. quadratisch oder kubisch).
#        - Wenn Sie **interaktive Effekte** zwischen den Merkmalen modellieren wollen.
#        - Wenn Ihre Daten nicht zu komplex sind und sich gut mit einem **Polynom** modellieren lassen.
#
#    - **Verwenden Sie den RBF Kernel (SVM_rbf)**:
#        - Wenn die Daten **hochdimensional** sind oder die Beziehung zwischen den Eingabedaten und der Zielgröße **nicht-linear** und komplex ist.
#        - Wenn Sie **keine genaue Vorstellung** von der Form der Beziehung haben und das Modell flexibel genug sein muss, um komplexe Muster zu erkennen.
#        - Wenn Ihre Daten **wenig strukturiert** sind und das Modell in einem nicht-linearen Raum trainiert werden soll, um komplexe Zusammenhänge zu erkennen.



# * Prophet Boost

# Das Modell ist besonders nützlich für Zeitreihenprognosen, 
# die sowohl saisonale Schwankungen (z.B. monatliche/jährliche Trends) 
# als auch Langzeittrends berücksichtigen müssen.
# Der Boosting-Ansatz kombiniert die Vorteile von Prophet 
# (saisonale und trendbasierte Modellierung) mit der Flexibilität und 
# Stärke von XGBoost, was zu besseren Vorhersagen bei komplexen und 
# großen Datensätzen führen kann.
# Verwendung: Wenn du historische Verkaufszahlen hast, die saisonale 
# und trendmäßige Muster aufweisen und du ein Modell benötigst, 
# das robuste Vorhersagen unter Berücksichtigung dieser Muster liefert.

Prophet_boost_model<- prophet_boost() %>%
  set_engine("prophet_xgboost") %>%
  fit(
    sales ~ date + as.numeric(date) + month(date, label = TRUE), 
    data = training(splits) 
  )

# * Arima Boost

# ARIMA ist besonders nützlich, wenn wir mit stationären Zeitreihen arbeiten, 
# bei denen der Trend und die saisonalen Effekte gut modelliert werden können.
# XGBoost wird verwendet, um die Leistung des ARIMA-Modells zu verbessern, 
# insbesondere wenn nicht-lineare Zusammenhänge in den Daten existieren.
# Verwendung:
#   Wenn wir historische Verkaufszahlen haben und sowohl saisonale Effekte 
# (z.B. Monat, Jahr) als auch langfristige Trends modellieren möchten.
# ARIMA alleine kann nur lineare Muster erfassen, aber die Kombination 
# mit XGBoost hilft dabei, auch komplexe und nicht-lineare Zusammenhänge in den Daten zu modellieren.

Arima_xgboost_model<- arima_boost()%>%
  set_engine("auto_arima_xgboost")%>%
  fit(
    sales~date + as.numeric(date)+ month(date, label = TRUE),
    data=training(splits)
  )



# 4 Modeltime Forecast Workflow----
model_tbl<- modeltime_table(
  arima_model,
  LM_model,
  Lm_model_no_trends,
  Prophet_model,
  RF_model,
  XgBoost_model,
  SVM_poly_model,
  SVM_rbf_model,
  Prophet_boost_model,
  Arima_xgboost_model
)


# * Calibration----
# run all models on the testing data 

calibration_tbl<- model_tbl %>%
  modeltime_calibrate(testing(splits))

#plot in an interactive format
calibration_tbl %>%
  modeltime_accuracy()%>%
  table_modeltime_accuracy(resizable=TRUE, bordered=TRUE)


# * Forecast and plot the results----
calibration_tbl%>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = clean_party,
    conf_interval = 0.95
  ) %>%
  plot_modeltime_forecast(.legend_show = TRUE,
                          .legend_max_width = 25)

# * Refit ----
# takes a Model time table and run the alg on a new data and fits parms

refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = clean_party) 

 forecast_tbl <- refit_tbl %>%
  modeltime_forecast(
    h = "1 year",
    actual_data = clean_party,
    conf_interval = 0.95
  ) 

forecast_tbl %>%
  plot_modeltime_forecast(.title = "Forecasted Sales-Refitted Models",.interactive = TRUE)


# 5 Model Averaging----
mean_forcast_tbl<- forecast_tbl %>%
  filter(.key !="actual")%>%
  group_by(.key,.index) %>% #group all predictions for one date 
  summarise(across(.value:.conf_hi,mean))%>% # summarize the value and conf and compute mean across all models
  mutate(
    .model_id=10,     # numb of models
    .model_desc="Average of Models"
  )

# * Visualization

forecast_tbl %>%
  filter(.key=="actual")%>%
  bind_rows(mean_forcast_tbl)%>%
  plot_modeltime_forecast()


# 5 Selecting Best ones -TRy----

#  Modeltime Forecast Workflow----
model_tbl<- modeltime_table(
  Prophet_model,
  Prophet_boost_model
)

# redo previous steps----
# * Calibration----
# run all models on the testing data 
calibration_tbl<- model_tbl %>%
  modeltime_calibrate(testing(splits))

#plot in an interactive format
calibration_tbl %>%
  modeltime_accuracy()%>%
  table_modeltime_accuracy(resizable=TRUE, bordered=TRUE)


# * Forecast and plot the results----
calibration_tbl%>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = clean_party,
    conf_interval = 0.9
  ) %>%
  plot_modeltime_forecast(.legend_show = TRUE,
                          .legend_max_width = 25)

# * Refit ----
# takes a Model time table and run the alg on a new data and fits parms

refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = clean_party) 

forecast_tbl <- refit_tbl %>%
  modeltime_forecast(
    h = "1 year",
    actual_data = clean_party,
    conf_interval = 0.80
  ) 

forecast_tbl %>%
  plot_modeltime_forecast(.interactive = TRUE)


# 5 Model Averaging----
mean_forcast_tbl<- forecast_tbl %>%
  filter(.key !="actual")%>%
  group_by(.key,.index) %>% #group all predictions for one date 
  summarise(across(.value:.conf_hi,mean))%>% # summarize the value and conf and compute mean across all models
  mutate(
    .model_id=2,     # numb of models
    .model_desc="Average of Models: Prophet & Prophet_Boost"
  )

# * Visualization

forecast_tbl %>%
  filter(.key=="actual")%>%
  bind_rows(mean_forcast_tbl)%>%
  plot_modeltime_forecast()

forecast_tbl$.value




# # Weekly Aggrgations
# Wöch_sales<- clean_party %>% 
#   summarise_by_time(
#     date, .by = "week",
#     weekly_sales=sum(sales),
#     weekly_promo=sum(onpromotion)
#   )
# 
# Wöch_sales %>% 
#   plot_time_series(
#     date, weekly_sales,
#     .title = "Wöchentliche Verkäufe- Store_9",
#     .interactive = TRUE
#   )




############################ Amelioration_step #######################################
# B Multivariate with Promotion----

# Retrain_models----
# Not all Models support Multivariate

# * Prophet
Prophet_model<- prophet_reg() %>%
  set_engine("prophet")%>%
  fit(
    sales~ date+onpromotion,
    data=training(splits)
  )

# * Prophet Boost
Prophet_boost_model<- prophet_boost() %>%
  set_engine("prophet_xgboost") %>%
  fit(
    sales ~ date + as.numeric(date) + month(date, label = TRUE)+ onpromotion, 
    data = training(splits) 
  )


# * LM
LM_model<- linear_reg()%>%
  set_engine("lm")%>%
  fit(
    sales ~ as.numeric(date) + month(date, label = TRUE)+ onpromotion, #as numeric for trends
    data=training(splits)
  )



model_tbl<- modeltime_table(
  Prophet_model,
  Prophet_boost_model,
  LM_model
  
)



# run all models on the testing data 
calibration_tbl<- model_tbl %>%
  modeltime_calibrate(testing(splits))

#plot in an interactive format
calibration_tbl %>%
  modeltime_accuracy()%>%
  table_modeltime_accuracy(resizable=TRUE, bordered=TRUE)


# * Forecast and plot the results----
calibration_tbl%>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = clean_party,
    conf_interval = 0.95
  ) %>%
  plot_modeltime_forecast(.legend_show = TRUE,
                          .legend_max_width = 25)

# * Refit ----
# takes a Model time table and run the alg on a new data and fits parms
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = clean_party) 

#test <- read_csv("Praxis/store-sales-time-series-forecasting/test.csv")


forecast_tbl <- refit_tbl %>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = clean_party,
    conf_interval = 0.95
  ) 

forecast_tbl %>%
  plot_modeltime_forecast(.interactive = TRUE)

# --> Best model is Prophet XGboost

# 5 Model Averaging----
# --> Not always a good solution just give a try
mean_forcast_tbl<- forecast_tbl %>%
  filter(.key !="actual")%>%
  group_by(.key,.index) %>% #group all predictions for one date 
  summarise(across(.value:.conf_hi,mean))%>% # summarize the value and conf and compute mean across all models
  mutate(
    .model_id=3,     # numb of models
    .model_desc="Average of Models"
  )

# * Visualization 

forecast_tbl %>%
  filter(.key=="actual")%>%
  bind_rows(mean_forcast_tbl)%>%
  plot_modeltime_forecast()

forecast_tbl$.value
