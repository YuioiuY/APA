import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

batch_size = 32
seq_len = 120
epochs = 30

def history_plot(history, title):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
    plt.title(f'{title}. График обучения')
    fig.gca().xaxis.get_major_locator().set_params(integer=True)
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Средняя ошибка')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(x), np.array(y)

def correlate(a, b):
    return np.corrcoef(a, b)[0, 1]

def show_predict(y_pred, y_true, dates, title=''):
    fig = plt.figure(figsize=(14, 7))
    plt.plot(dates[1:], y_pred[1:], label='Прогноз')
    plt.plot(dates[:-1], y_true[:-1], label='Базовый')
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_corr(y_pred, y_true, title='', break_step=30):
    y_len = y_true.shape[0]
    steps = range(1, np.min([y_len+1, break_step+1]))
    cross_corr = [correlate(y_true[:-step, 0], y_pred[step:, 0]) for step in steps]
    auto_corr = [correlate(y_true[:-step, 0], y_true[step:, 0]) for step in steps]
    plt.plot(steps, cross_corr, label='Прогноз')
    plt.plot(steps, auto_corr, label='Эталон')
    plt.title(title)
    plt.xticks(steps)
    plt.xlabel('Шаги смещения')
    plt.ylabel('Коэффициент корреляции')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_pred(model, x_test, y_test, y_scaler):
    y_pred_unscaled = y_scaler.inverse_transform(model.predict(x_test, verbose=0))
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    return y_pred_unscaled, y_test_unscaled

file_path = 'AAPL.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.set_index('Date', inplace=True)

close_prices = data['Close']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

x, y = create_sequences(scaled_data, seq_len)
split = int(len(x) * 0.8)
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, 1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

history_plot(history, 'LSTM(64+32) модель')

y_pred, y_true = get_pred(model, x_test, y_test, scaler)

test_dates = data.index[-len(x_test):]
show_predict(y_pred, y_true, dates=test_dates, title='Сравнение прогноза и базового ряда')

show_corr(y_pred, y_true, title='Корреляция прогноза и базового ряда')
plot_acf(y_true.flatten(), lags=40, title='ACF реального ряда')
plt.show()
plot_acf(y_pred.flatten(), lags=40, title='ACF прогноза')
plt.show()
plot_acf((y_true - y_pred).flatten(), lags=40, title='ACF ошибки')
plt.show()
