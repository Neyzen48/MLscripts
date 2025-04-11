
# Amelioration using holidays----


holidays_events <- read_csv("Praxis/store-sales-time-series-forecasting/holidays_events.csv")

Party_holidays_promo<- Party_df %>% 
  left_join(holidays_events, by="date") %>% 
  mutate(holiday_flag= ifelse(is.na(type),0,1))


splits<- time_series_split( Party_holidays_promo, assess = "1 year", cumulative = TRUE)

# Retraining Models----

# * Prophet
Prophet_model<- prophet_reg() %>%
  set_engine("prophet")%>%
  fit(
    sales~ date+onpromotion+holiday_flag,
    data=training(splits)
  )

# * Prophet Boost
Prophet_boost_model<- prophet_boost() %>%
  set_engine("prophet_xgboost") %>%
  fit(
    sales ~ date + as.numeric(date) + month(date, label = TRUE)+ onpromotion+holiday_flag, 
    data = training(splits) 
  )


# * LM
LM_model<- linear_reg()%>%
  set_engine("lm")%>%
  fit(
    sales ~ as.numeric(date) + month(date, label = TRUE)+ onpromotion+holiday_flag, #as n umeric for trends
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
    actual_data = Party_holidays_promo,
    conf_interval = 0.95
  ) %>%
  plot_modeltime_forecast(.legend_show = TRUE,
                          .legend_max_width = 25)

# * Refit ----
# takes a Model time table and run the alg on a new data and fits parms
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = Party_holidays_promo) 

#test <- read_csv("Praxis/store-sales-time-series-forecasting/test.csv")


forecast_tbl <- refit_tbl %>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = Party_holidays_promo,
    conf_interval = 0.95
  ) 

forecast_tbl %>%
  plot_modeltime_forecast(.interactive = TRUE)



# 5 Model Averaging----  No Amelioration--> worst 
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

