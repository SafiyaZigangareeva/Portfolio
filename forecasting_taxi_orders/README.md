#  Прогнозирование заказов такси

## Описание проекта и цель

Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час на основании исторических данных о заказах такси в аэропортах. Постройте модель для такого предсказания.

Значение метрики RMSE на тестовой выборке должно быть не больше 48.

## Описание данных

Количество заказов находится в столбце `num_orders`

## План работы

 - Загрузить данные и выполнить их ресемплирование по одному часу.
 - Проанализировать данные.
 - Обучить разные модели с различными гиперпараметрами. 
 - Проверить данные на тестовой выборке и сделать выводы.

## Выводы

Для решения задачи предсказания количества заказов на следующий час в такси использовался временной ряд (ВР), который представляет собой временной отрезок в 183 дня с интервалом в 10 мин. Критерий Дики-Фуллера составил 0.03, поэтому ВР будем считать стационарным.

После ресемплирования данных выявлено два периода сезонности - сутки и неделя. На основании этого были созданы новые признаки для обучения модели.

Константная модель (предсказывает всем наблюдениям среднее значение) показала результат RMSE 84.

Для обучения использовались модели:
 - LinearRegression, результат на тренировочной выборке RMSE 22.45
 - RandomForestRegressor, результат на тренировочной выборке RMSE 22.82
 - LGBMRegressor, результат на тренировочной выборке RMSE 22.72  
 
Лучший результат у модели LinearRegression, ее результат на тестовых данных RMSE 34.47
На графиках сравнения предсказаний модели и реальных данных видно, что модель достаточно хорошо описывает данные, сглаживая пики.