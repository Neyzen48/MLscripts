


# Amelioration using Weather----

# - reim package: http://ropensci.github.io/riem/index.html
# - Map: https://mesonet.agron.iastate.edu/request/download.phtml?network=VA_ASOS

# VA Weather Stations, Use DCA (See Map) 
# here i took example city Guayaquil:
#https://mesonet.agron.iastate.edu/request/daily.phtml?network=EC__ASOS
# https://mesonet.agron.iastate.edu/sites/site.php?network=EC__ASOS&station=SEGU



## Denken Sie an Wechselwirkungseffekte und andere Kodierungen --> Verbesserungsvorschläge 


riem::riem_stations("EC__ASOS") 
weather_Guayaquil_tbl <- riem::riem_measures("SEGU", date_start = "2013-01-01", date_end = "2018-01-01")
weather_Guayaquil_tbl 


# Cleaning
library(zoo) # interpolation impute

weather_avg <- weather_Guayaquil_tbl %>%
  mutate(valid = as.Date(substr(valid, 1, 10))) %>% 
  # Convert tmp into Celsius
  mutate(temperature_cel = (feel - 32) * (5 / 9)) %>% 
  group_by(valid) %>% 
  summarise(
    avg_temp = mean(temperature_cel, na.rm = TRUE)
  ) %>% 
  # Complete the sequence of dates to ensure that there's a row for every date in the dataset before performing interpolation.
  complete(valid = seq(min(valid), max(valid), by = "day")) %>%
  # Interpolate missing values using linear interpolation
  mutate(avg_temp = na.approx(avg_temp))%>% 
  rename("date"="valid")


# Bind Datasets
Holiday_weather_tbl<- Party_holidays_promo %>% 
  left_join(weather_avg, by="date")

splits<- time_series_split( Holiday_weather_tbl, assess = "1 year", cumulative = TRUE)


# Retraining Models----
# * Prophet
Prophet_model<- prophet_reg() %>%
  set_engine("prophet")%>%
  fit(
    sales~ date+onpromotion+holiday_flag+avg_temp,
    data=training(splits)
  )

# * Prophet Boost
Prophet_boost_model<- prophet_boost() %>%
  set_engine("prophet_xgboost") %>%
  fit(
    sales ~ date + as.numeric(date) + month(date, label = TRUE)+ onpromotion+holiday_flag+ avg_temp, 
    data = training(splits) 
  )


# * LM
LM_model<- linear_reg()%>%
  set_engine("lm")%>%
  fit(
    sales ~ as.numeric(date) + month(date, label = TRUE)+ onpromotion+holiday_flag+avg_temp, #as numeric for trends
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
    actual_data = Holiday_weather_tbl,
    conf_interval = 0.95
  ) %>%
  plot_modeltime_forecast(.legend_show = TRUE,
                          .legend_max_width = 25)

# * Refit ----
# takes a Model time table and run the alg on a new data and fits parms
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = Holiday_weather_tbl) 


forecast_tbl <- refit_tbl %>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = Holiday_weather_tbl,
    conf_interval = 0.95
  ) 

forecast_tbl %>%
  plot_modeltime_forecast(.interactive = TRUE)

# Prophet with XgBoost is the best model to apply.

### !!! When the TS curve is having spikes then averaging does not make any sense
# 5 Model Averaging---- Not always a good solution --> here also worst
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



# Final Forecast----

sample_data <- read_csv("TS_BSP_PARTY_store9/sample_data.csv")

Future_sales <- refit_tbl %>%
  modeltime_forecast(
    new_data = sample_data,
    actual_data = Holiday_weather_tbl,
    conf_interval = 0.95
  ) 

Future_sales %>%
  plot_modeltime_forecast(.interactive = TRUE)
