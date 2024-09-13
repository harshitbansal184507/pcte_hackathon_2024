import streamlit as st
import altair as alt 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
from sklearn import preprocessing
import pydeck as pdk
import random 
import seaborn as sns 

data= pd.read_csv("data/diabetes2.csv")
specializations = ['Endocrinologist', 'General Physician', 'Diabetologist', 'Pediatrician', 'Internal Medicine']
data['specialization'] = [random.choice(specializations) for _ in range(len(data))]
experience_range = range(1, 41)
data['experience'] = [random.choice(experience_range) for _ in range(len(data))]
df = pd.DataFrame(data)


st.set_page_config(
    page_title="Doctor Preferences for Lyumjev Solution",
    page_icon="images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title("Doctor Preferences for Lyumjev Solution")
    st.image("images/logo.png")  
    Home = st.button("Home", use_container_width=True)

    market_analysis = st.button("Market Analysis", use_container_width=True)
    zone_specific = st.button("Zone-specific Metrics", key="Zone_Specific", use_container_width=True)
    doctor_segmentation = st.button("Doctor Behavioral Segmentation", use_container_width=True)
    feedback_vs_volume = st.button("Feedback vs Prescription Volume", use_container_width=True)
    demographic_insights = st.button("Lyumjev Adoption Heatmap", use_container_width=True)
    sales_performance = st.button("Sales Performance", use_container_width=True)
    recommendation_report = st.button("Recommendation Report", use_container_width=True)

if zone_specific:
    st.title("Zone-specific Metrics")
    st.write("Detailed metrics by geographic zone for Lyumjev solution")
    
    north = [
    'Haryana', 
    'Himachal Pradesh', 
    'Punjab', 
    'Rajasthan', 
    'Uttar Pradesh', 
    'Uttarakhand', 
    'Chandigarh', 
    'Delhi'
]
    south = [
    'Andhra Pradesh', 
    'Karnataka', 
    'Kerala', 
    'Tamil Nadu', 
    'Telangana', 
    'Goa', 
    'Lakshadweep', 
    'Puducherry', 
    'Andaman and Nicobar Islands'
]
    east = [
    'Arunachal Pradesh', 
    'Assam', 
    'Bihar', 
    'Chhattisgarh', 
    'Jharkhand', 
    'Odisha', 
    'Sikkim', 
    'West Bengal'
]
    west = [
    'Gujarat', 
    'Maharashtra', 
    'Rajasthan', 
    'Dadra and Nagar Haveli and Daman and Diu'
]
    regions = ["North", "South","East", "West"]
    counts= {"Lyumjev": np.array([
        data[(data['state'].isin(north)) & (data['insulin_name'] == "Lyumjev")].shape[0],
        data[(data['state'].isin(south)) & (data['insulin_name'] == "Lyumjev")].shape[0],
        data[(data['state'].isin(east)) & (data['insulin_name'] == "Lyumjev")].shape[0],
        data[(data['state'].isin(west)) & (data['insulin_name'] == "Lyumjev")].shape[0]
    ]),
    "Humalog": np.array([
        data[(data['state'].isin(north)) & (data['insulin_name'] == "Humalog")].shape[0],
        data[(data['state'].isin(south)) & (data['insulin_name'] == "Humalog")].shape[0],
        data[(data['state'].isin(east)) & (data['insulin_name'] == "Humalog")].shape[0],
        data[(data['state'].isin(west)) & (data['insulin_name'] == "Humalog")].shape[0]
    ]),
    "Novolog": np.array([
        data[(data['state'].isin(north)) & (data['insulin_name'] == "Novolog")].shape[0],
        data[(data['state'].isin(south)) & (data['insulin_name'] == "Novolog")].shape[0],
        data[(data['state'].isin(east)) & (data['insulin_name'] == "Novolog")].shape[0],
        data[(data['state'].isin(west)) & (data['insulin_name'] == "Novolog")].shape[0]
    ])}
    width=0.5
    fig, ax = plt.subplots()
    bottom = np.zeros(len(regions))
    
    for insulin , insulin_count in counts.items():
        p=ax.bar(regions,insulin_count , width , label = insulin , bottom=bottom)
        bottom+=insulin_count
        ax.bar_label(p, label_type='center')
             
    ax.set_title('Region wise Insulin Data')
    ax.legend()
    
    st.pyplot(fig)
    
    
    
elif market_analysis:
    st.title("Market Analysis")
    st.write("The Following graph shows market share by Lyumjev and other Insulin Competitors")
    state_to_region = {
    'Delhi': 'North', 'Haryana': 'North', 'Punjab': 'North', 'Uttar Pradesh': 'North',
    'Rajasthan': 'North', 'Uttarakhand': 'North', 'Himachal Pradesh': 'North',
    'West Bengal': 'East', 'Bihar': 'East', 'Odisha': 'East', 'Jharkhand': 'East',
    'Assam': 'East', 'Sikkim': 'East', 'Arunachal Pradesh': 'East',
    'Kerala': 'South', 'Tamil Nadu': 'South', 'Karnataka': 'South', 'Andhra Pradesh': 'South',
    'Telangana': 'South', 'Lakshadweep': 'South',
    'Maharashtra': 'West', 'Gujarat': 'West', 'Goa': 'West',
    'Chhattisgarh': 'Central', 'Madhya Pradesh': 'Central', 'Jharhand': 'Central'
}
    data['region'] = data['state'].map(state_to_region)

    total_prescription_volume_by_region = data.groupby('region')['prescription_volume'].sum().reset_index()
    total_prescription_volume_by_region.rename(columns={'prescription_volume': 'total_volume'}, inplace=True)

    data = pd.merge(data, total_prescription_volume_by_region, on='region')

    data['percentage'] = data.apply(
        lambda row: (row['prescription_volume'] / row['total_volume']) * 100, axis=1
    )

    lyumjev_data = data[data['insulin_name'] == 'Lyumjev']

    lyumjev_percentage = lyumjev_data.groupby('region')['percentage'].sum().reset_index()
    lyumjev_percentage.rename(columns={'percentage': 'lyumjev_percentage'}, inplace=True)

    other_insulins_data = data[data['insulin_name'] != 'Lyumjev']
    other_insulins_percentage = other_insulins_data.groupby('region')['percentage'].sum().reset_index()
    other_insulins_percentage.rename(columns={'percentage': 'other_percentage'}, inplace=True)

    merged_summary = pd.merge(lyumjev_percentage, other_insulins_percentage, on='region')

    plt.figure(figsize=(10, 6))

    plt.bar(merged_summary['region'], merged_summary['lyumjev_percentage'], color='skyblue', label='Lyumjev')

    plt.bar(merged_summary['region'], merged_summary['other_percentage'], 
            bottom=merged_summary['lyumjev_percentage'], color='lightcoral', label='Other Insulins')

    plt.xlabel('Region')
    plt.ylabel('Market Share Percentage')
    plt.title('Lyumjev Market Share by Region')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(plt)
    
    
    
    
    
    
    
    
elif doctor_segmentation:
    st.title("Doctor Behavioral Segmentation")
    st.write("Segmentation of doctors based on their behavior and preferences.")
    col1 , col2 , col3 = st.columns(spec=3)
    x=[((data['feedback'] == 'Positive') & (data['insulin_name'] == 'Lyumjev')).sum() , 
       ((data['feedback'] == 'Negative') & (data['insulin_name'] == 'Lyumjev')).sum() , 
       ((data['feedback'] == 'Neutral') & (data['insulin_name'] == 'Lyumjev')).sum()





]
    fig, ax = plt.subplots(figsize=(6, 4))
    Labels =["Positive", "Negative", "Neutral"]
    ax.pie(x,labels=Labels,  radius=2, center=(4, 4),
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True, labeldistance=1.2)
    
    

    

    with col1 , col2 :
        st.pyplot(fig)









elif feedback_vs_volume:
    st.title("Feedback vs Prescription Volume")
    st.write("Correlation between feedback received and prescription volume.")
    
    label_encoder = preprocessing.LabelEncoder() 
    data["feedback"]=label_encoder.fit_transform(data["feedback"])
    
    x=data["prescription_volume"]
    y=data["feedback"]
    fig , ax = plt.subplots()
    ax.scatter(x, y, s=80, alpha=0.8, edgecolors="k")
    b, a = np.polyfit(x, y, deg=1)  
    xseq = np.linspace(0, 10, num=100) 
    ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
    st.pyplot(fig)

    


    
    






elif demographic_insights:
    data['count'] = data.groupby('state')['state'].transform('count')

    st.title("Lyumjev Adoption Heatmap: Specialization & Regional Insights")
    north = [
    'Haryana', 'Himachal Pradesh', 'Punjab', 'Rajasthan', 
    'Uttar Pradesh', 'Uttarakhand', 'Chandigarh', 'Delhi'
]
    south = [
        'Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 
        'Telangana', 'Goa', 'Lakshadweep', 'Puducherry', 'Andaman and Nicobar Islands'
    ]
    east = [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 
        'Jharkhand', 'Odisha', 'Sikkim', 'West Bengal'
    ]
    west = [
        'Gujarat', 'Maharashtra', 'Rajasthan', 
        'Dadra and Nagar Haveli and Daman and Diu'
    ]

    def get_region(state):
        if state in north:
            return 'North'
        elif state in south:
            return 'South'
        elif state in east:
            return 'East'
        elif state in west:
            return 'West'
        

    df['region'] = df['state'].apply(get_region)

    lyumjev_df = df[df['insulin_name'] == 'Lyumjev']

    heatmap_data = lyumjev_df.groupby(['specialization', 'region'])['prescription_volume'].sum().unstack().fillna(0)

    st.write('Heatmap of Lyumjev Adoption by Specialization and Region in India')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=.5, ax=ax)

    ax.set_title('Heatmap of Lyumjev Adoption by Specialization and Region in India')
    ax.set_xlabel('Region')
    ax.set_ylabel('Specialization')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)
    


    
    

    
elif sales_performance:
    st.title("Sales Performance")
    st.write("Sales performance data and trends")
    
    state_to_region = {
        'Jammu & Kashmir': 'North', 'Punjab': 'North', 'Haryana': 'North', 'Himachal Pradesh': 'North', 'Uttarakhand': 'North', 'Delhi': 'North',
        'Uttar Pradesh': 'North', 'Rajasthan': 'North', 'Bihar': 'East', 'West Bengal': 'East', 'Odisha': 'East',
        'Assam': 'East', 'Sikkim': 'East', 'Arunachal Pradesh': 'East', 'Nagaland': 'East', 'Manipur': 'East', 'Mizoram': 'East',
        'Tripura': 'East', 'Meghalaya': 'East', 'Gujarat': 'West', 'Maharashtra': 'West', 'Goa': 'West',
        'Madhya Pradesh': 'West', 'Chhattisgarh': 'West', 'Jharkhand': 'East',
        'Karnataka': 'South', 'Tamil Nadu': 'South', 'Kerala': 'South', 'Andhra Pradesh': 'South', 'Telangana': 'South'
    }

    lyumjev_data = df[df['insulin_name'] == 'Lyumjev'].copy()

    lyumjev_data.loc[:, 'year'] = lyumjev_data['year'].astype(int)

    lyumjev_data['region'] = lyumjev_data['state'].map(state_to_region)

    unique_years = lyumjev_data['year'].unique()

    for year in unique_years:
        yearly_data = lyumjev_data[lyumjev_data['year'] == year]
        
        regional_sales = yearly_data.groupby('region').agg({
            'prescription_volume': 'sum'
        }).reset_index()
        
        fig = px.bar(regional_sales, x='region', y='prescription_volume',
                    title=f'Total Lyumjev Prescriptions by Region in {year}',
                    labels={'region': 'Region', 'prescription_volume': 'Total Prescription Volume'},
                    color='region',  # Color bars by region
                    color_discrete_sequence=px.colors.qualitative.Plotly)  # Use Plotly color palette
        
        fig.update_layout(xaxis_title='Region', yaxis_title='Total Prescription Volume', xaxis_tickangle=-45)


        st.plotly_chart(fig)
elif recommendation_report:
    st.title("Recommendation Report")
    st.write("Final recommendations based on the analysis")
    
    
    if 'region' not in data.columns:
        region_mapping = {
            'North': ['Delhi', 'Haryana', 'Punjab', 'Uttarakhand', 'Uttar Pradesh'],
            'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu'],
            'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
            'West': ['Goa', 'Gujarat', 'Maharashtra', 'Rajasthan']
        }
        data['region'] = data['state'].apply(lambda x: next((r for r, states in region_mapping.items() if x in states), 'Unknown'))

    data = data[data['region'] != 'Unknown']

    lyumjev_data = data[data['insulin_name'] == 'Lyumjev']

    region_adoption = lyumjev_data.groupby('region')['prescription_volume'].sum().reset_index()

    


    insulin_comparison = data.groupby(['insulin_name', 'region'])['prescription_volume'].sum().unstack().fillna(0)

    st.write("### Comparison of Insulin Adoption by Region")
    fig, ax = plt.subplots(figsize=(12, 8))
    insulin_comparison.plot(kind='bar', stacked=True, colormap='tab20', ax=ax)
    ax.set_title('Comparison of Insulin Adoption by Region')
    ax.set_xlabel('Region')
    ax.set_ylabel('Total Prescription Volume')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Insulin Type')
    st.pyplot(fig)
    
    st.header("Recommendations")
    st.write("The report indicates that while the overall insulin market is more substantial in the southern regions of India, the adoption rates for Lyumjev insulin are notably higher in the northern regions. This suggests that Cipla should consider the southern parts of India as a promising market for expanding its presence and increasing Lyumjev’s adoption. ")


    if 'doctor_specialty' in data.columns:
        lyumjev_data = lyumjev_data[lyumjev_data['doctor_specialty'] != 'Unknown']
        
        doctor_behavior = lyumjev_data.groupby('doctor_specialty')['prescription_volume'].sum().reset_index()

        st.write("### Lyumjev Adoption by Doctor Specialty")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=doctor_behavior, x='doctor_specialty', y='prescription_volume', hue='doctor_specialty', palette='coolwarm', legend=False, ax=ax)
        ax.set_title('Lyumjev Adoption by Doctor Specialty')
        ax.set_xlabel('Doctor Specialty')
        ax.set_ylabel('Total Prescription Volume')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    


    
elif Home:
    st.title("Diabetes In India")
    col1 , col2 , col3= st.columns(spec=3 )

    with col1 :
        st.header("Total Diabetic Patients in India", divider="red")
        st.header("110 Million")
    with col2 :
        st.header("Growth Rate of Diabetes in India", divider="red")
        st.header("10.4% by 2030")    
    with col3 :
        st.header("Insulin Market in India as of 2023", divider="red")
        st.header("US $4.2 Billion")  
    st.image("images/home_image.jpg", use_column_width=True)          
        
        
        
    
    
        
    
    
