import pandas as pd
import statsmodels.api as sm

data = pd.read_csv(r"/Users/deeksha_dwarakanath/Downloads/Resume Projects/Linear Regression - Housing Prices/housing_price_dataset.csv", index_col=False, header=0)
list_of_headers = list(data.columns.values)
list_of_headers_copy = list(data.columns.values)
num = 1

def read_variables(h_list):
    global num
    success1 = False

    while not success1:
        try:
            f_var = input(f"{num} variable {h_list}: ")
            h_list.remove(f_var)
            success1 = True
            num += 1
            return f_var
        except ValueError:
            print("I'm sorry, that header does not exist within the dataframe.")

first_variable = read_variables(list_of_headers_copy)
second_variable = read_variables(list_of_headers_copy)
del num
del list_of_headers_copy

x = data[first_variable].values.reshape(-1, 1)
y = data[second_variable].values.reshape(-1, 1)
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())
