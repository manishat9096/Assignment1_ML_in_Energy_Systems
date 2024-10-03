from Step_2_Victor import *

def gaussian(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

def epanechnikov(t):
    res = np.zeros_like(t)
    res[np.abs(t) <= 1] = 0.75 * (1 - t[np.abs(t) <= 1]**2)
    return res

def tricube(t):
    res = np.zeros_like(t)
    res[np.abs(t) <= 1] = (70 / 81) * (1 - np.abs(t[np.abs(t) <= 1])**3)**3
    return res

def uniform(t, p=0.2):
    return np.zeros_like(t) + p

def triangle(t):
    res = np.zeros_like(t)
    res[np.abs(t) <= 1] = 1 - np.abs(t[np.abs(t) <= 1])
    return res

fig, ax = plt.subplots(figsize=(9, 5))

xs = np.linspace(-1.5, 1.5, 1000)
for func in [gaussian, epanechnikov, tricube, uniform, triangle]:
    ax.plot(xs, func(xs), label=func.__name__)

ax.legend()
ax.set_ylabel("$K_{\lambda}(x_0, x)$")
ax.set_xlabel("$||x_0 - x||_2$")
fig.canvas.draw()


X_normalized_new = X_normalized.copy()  
X_normalized_new.insert(0, 'intercept', 1) 

X_WLS = X_normalized_new
y_WLS = y_normalized

interval_fitting_points = 20
X_query = X_WLS.iloc[::interval_fitting_points]
y_comp = y_normalized.iloc[::interval_fitting_points]


def weighted_least_squares(X_query, X_WLS, radius, tau):
    y_pred_wls = np.zeros(len(X_query))
    for i in range(len(X_query)):
        W = np.diagflat(gaussian(np.linalg.norm(X_WLS - X_query.iloc[i], axis=1) / radius))
        theta = np.dot(np.linalg.inv(np.dot(X_WLS.T, np.dot(W, X_WLS))), np.dot(X_WLS.T, np.dot(W, y_WLS)))
        y_pred_wls[i] = np.dot(X_query.iloc[i], theta)
    return y_pred_wls

tau = 0.05
radius_values = np.linspace(0.15, 1.0, 10)
rmse_values = []

for radius in radius_values:
    print("Radius :", radius)
    y_pred = weighted_least_squares(X_query, X_WLS, radius, tau)
    rmse_values.append(np.sqrt(mean_squared_error(y_comp, y_pred)))
    
plt.figure(figsize=(14, 7), dpi = 300)
plt.plot(radius_values, rmse_values, label='RMSE vs radius', linewidth=1)


best_radius = radius_values[np.argmin(rmse_values)]

interval_fitting_points = 1
X_query = X_WLS.iloc[::interval_fitting_points]
y_comp = y_normalized.iloc[::interval_fitting_points]

y_pred_best = weighted_least_squares(X_query, X_WLS, best_radius, tau)

mse_nlr = mean_squared_error(y_comp, y_pred_best)
rmse_nlr = np.sqrt(mse_nlr)
print(f"Test RMSE w/ NLR: {rmse_nlr:0.10f}")
mae_nlr = mean_absolute_error(y_comp, y_pred_best)
print(f"Test MAE w/ NLR: {mae_nlr:0.10f}")
 
plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_comp.index, y_comp, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions gradient desscent
plt.scatter(y_comp.index, y_pred_best, color='blue', alpha=0.6, label='Predicted Values', s=50)
plt.show()

