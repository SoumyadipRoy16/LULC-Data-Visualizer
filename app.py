from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

excel_file_path = r'LULC DATA.xlsx'

df_sheet1 = pd.read_excel(excel_file_path, sheet_name='2015-16')
df_sheet2 = pd.read_excel(excel_file_path, sheet_name='2005-06')
df_sheet3 = pd.read_excel(excel_file_path, sheet_name='2011-12')

df_sheet1 = df_sheet1.dropna(axis=1, how='all')
df_sheet2 = df_sheet2.dropna(axis=1, how='all')
df_sheet3 = df_sheet3.dropna(axis=1, how='all')

df_sheet1 = df_sheet1.dropna(axis=0, how='all')
df_sheet2 = df_sheet2.dropna(axis=0, how='all')
df_sheet3 = df_sheet3.dropna(axis=0, how='all')

wms_url_dict = {
    '2015-16': 'https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=lulc:{state}_LULC50K_1516&FORMAT=image/png&TRANSPARENT=true&WIDTH=800&HEIGHT=800&SRS=EPSG:4326&BBOX={bbox}',
    '2005-06': 'https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=lulc:{state}_LULC50K_0506&FORMAT=image/png&TRANSPARENT=true&WIDTH=800&HEIGHT=800&SRS=EPSG:4326&BBOX={bbox}',
    '2011-12': 'https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=lulc:{state}_LULC50K_1112&FORMAT=image/png&TRANSPARENT=true&WIDTH=800&HEIGHT=800&SRS=EPSG:4326&BBOX={bbox}',

}

list_of_states = ["UP", "UK", "UT", "AP", "AR", "AS", "BR", "CH", "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "WB"]


def train_linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to calculate correlation between LULC and GDP using the trained model
def calculate_correlation_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return r2

df = pd.read_excel(excel_file_path, sheet_name='Test')
X = df[['Agriculture', 'Barren/Unculturable Wastelands', 'Builtup', 'Forest', 'Grass/Grazing', 'Snow and Glacier', 'Wet lands/Waterbodies']]  # Replace with actual LULC columns
y = df['GDP']  # Replace with actual GDP column

# Train the linear regression model
linear_model = train_linear_regression_model(X, y)

r2_score = calculate_correlation_model(linear_model, X, y)
coefficients = linear_model.coef_
feature_names = X.columns
influential_attributes = {feature: coefficient for feature, coefficient in zip(feature_names, coefficients)}

    
@app.route('/', methods=['GET', 'POST'])
def index1():
    if request.method == 'POST':
        state_input = request.form['state_input']
        graph_type = request.form['graph_type']
        selected_year = request.form['selected_year']

        if selected_year == '2015-16':
            selected_data = df_sheet1
        elif selected_year == '2005-06':
            selected_data = df_sheet2
        elif selected_year == '2011-12':
            selected_data = df_sheet3
        else:
            return render_template('index1.html', error_msg='Invalid year selected! Try again')
           

        if state_input in list_of_states:

            bbox_values = {
                'AP': '76.761,12.624,84.765,19.917',
                'AR': '91.605,26.656,97.415,29.376',
                'AS': '89.701,24.135,96.021,27.977',
                'BR': '83.323,24.286,88.298,27.521',
                'CH': '80.246,17.782,84.396,24.105',
                'GA': '73.676,14.9,74.336,15.801',
                'GJ': '68.148,20.121,74.477,24.714',
                'HR': '74.475,27.653,77.593,30.929',
                'HP': '75.595,30.377,79.012,33.266',
                'JK': '73.799,32.28,79.603,35.23',
                'JH': '83.33,21.97,87.962,25.349',
                'KA': '74.054,11.592,78.588,18.45',
                'KL': '74.862,8.289,77.413,12.795',
                'MP': '74.031,21.073,82.816,26.871',
                'MH': '72.643,15.606,80.898,22.027',
                'MN': '92.974,23.843,94.747,25.968',
                'ML': '89.822,25.032,92.804,26.119',
                'MZ': '92.259,21.948,93.438,24.521',
                'NL': '93.332,25.202,95.245,27.043',
                'OR': '81.34,17.794,87.486,22.568',
                'PB': '73.881,29.544,76.943,32.511',
                'RJ': '69.482,23.062,78.272,30.198',
                'SK': '88.012,27.082,88.922,28.131',
                'TN': '76.234,8.075,80.349,13.565',
                'TS': '77.16,15.46,81.43,19.47',
                'TR': '91.4988,23.5204,92.7622,24.5347',
                'UK': '77.575,28.715,81.043,31.467',
                'UP': '77.084,23.87,84.634,30.408', 
                'WB': '85.82,21.481,89.886,27.22',  
            }
            
            
            if state_input in list_of_states and state_input in bbox_values:
                bbox = bbox_values[state_input]

                wms_url = wms_url_dict[selected_year].format(state=state_input, bbox=bbox)
                           
            selected_index1es_lulc = [0, 6, 13, 17, 22, 24, 27]
            selected_index1es_gdp = [4, 11, 15, 21, 23, 25, 30]

            plt.figure(figsize=(10, 8))

            if graph_type == 'line':
                plt.plot(selected_data.loc[selected_index1es_lulc, 'LULC/STATES(UT)'], selected_data.loc[selected_index1es_gdp, state_input])
                plt.xlabel('LULC')
                plt.ylabel(f'GDP ({selected_year}) - {state_input}')
                plt.title(f'{state_input} - Line Graph')
                
                
            else:
                plt.pie(selected_data.loc[selected_index1es_gdp, state_input], labels=selected_data.loc[selected_index1es_lulc, 'LULC/STATES(UT)'], autopct='%1.1f%%')
                plt.title(f'GDP Distribution for {state_input}')

            plt.tight_layout()

            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)

            plot_url = base64.b64encode(img.getvalue()).decode()

            plt.close() 
            
            legend_data = {"Builtup, Urban": "#FF0000", "Builtup, Rural": "#964B00", "Builtup, Mining": "#C4A484", "Agriculture, Crop Land": "#FAFA33", "Agriculture, Plantation": "#FFE200", "Agriculture, Fallow": "#FEFEB1", "Forest,Evergreen / Semi Evergreen": "#043927", "Forest, Deciduos": "#50D300", "Forest, Forest Plantation": "#44B200", "Forest, Scrub Forest": "#A7DFB3", "Forest, Swamp/Mangroves": "#11FFEE", "Grass/Grazing": "#AFCB80", "Barren/unculturable/Wastelands,Scrub land": "#FF10F0", "Barren/unculturable/Wastelands, Sandy area": "#EBE8FC", "Barren/unculturable/Wastelands, Barren rocky": "#FFB6C1", "Wetlands/Waterbodies, Inland Wetland": "#39AD48", "Wetlands/Waterbodies, Coastal Wetland": "#39FF14", "Wetlands/Waterbodies, River/Stram/Canals": "#002E5C", "Wetlands/Waterbodies, Reservoir/Lakes/Ponds": "#729FE0", "Snow and Glaciers": "#D3D3D3"}  

            return render_template('index1.html', wms_url=wms_url, plot_url=plot_url, list_of_states=list_of_states, state_input=state_input, selected_year=selected_year, legend_data=legend_data, graph_type=graph_type, r2_score=r2_score, influential_attributes=influential_attributes)
        else:
            return render_template('index1.html', error_msg='Invalid state code entered! Try again')

    return render_template('index1.html', list_of_states=list_of_states)

if __name__ == '__main__':
    app.run(debug=True)
