
import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import os
import gdown
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import streamlit as st

def create_stylish_line_plot(df, x_col, y1_col, y2_col, title="Stylish Line Plot", x_label="X", y_label="Values"):
    """
    Membuat line plot yang menarik dengan dua kolom y berbeda dan kolom x sebagai sumbu x.

    Parameters:
    - df: DataFrame yang berisi data.
    - x_col: Nama kolom yang akan digunakan sebagai sumbu x.
    - y1_col: Nama kolom yang akan digunakan sebagai garis pertama.
    - y2_col: Nama kolom yang akan digunakan sebagai garis kedua.
    - title: Judul plot.
    - x_label: Label untuk sumbu x.
    - y_label: Label untuk sumbu y.
    """
    
    # Membuat trace untuk y1
    trace1 = go.Scatter(
        x=df[x_col],
        y=df[y1_col],
        mode='lines+markers',
        name=f'{y1_col}',
        line=dict(color='dodgerblue', width=2),
        marker=dict(size=8)
    )

    # Membuat trace untuk y2
    trace2 = go.Scatter(
        x=df[x_col],
        y=df[y2_col],
        mode='lines+markers',
        name=f'{y2_col}',
        line=dict(color='orange', width=2),
        marker=dict(size=8)
    )

    # Membuat layout untuk plot
    layout = go.Layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color='darkblue')),
        xaxis=dict(title=x_label, titlefont=dict(size=16, color='darkblue')),
        yaxis=dict(title=y_label, titlefont=dict(size=16, color='darkblue')),
        showlegend=True,
        legend=dict(font=dict(size=12), x=0, y=1, xanchor='left', yanchor='top'),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray',
        shapes=[
            # Garis putus-putus merah di y=0.5
            dict(
                type="line",
                x0=df[x_col].min(), x1=df[x_col].max(),
                y0=0.5, y1=0.5,
                line=dict(color="red", width=1, dash="dash")
            )
        ]
    )

    # Membuat figure dari trace dan layout
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Menampilkan plot di Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
st.set_page_config(layout="wide")

def add_min_width_css():
    st.markdown(
        """
        <style>
        /* Set a minimum width for the app */
        .css-1d391kg {
            min-width: 3000px; /* Set the minimum width */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add CSS styling to the app
add_min_width_css()

def download_file_from_github(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def load_excel(file_path):
    with open(file_path, 'rb') as file:
        model = pd.read_excel(file, engine='openpyxl')
    return model

def list_files_in_directory(dir_path):
    # Fungsi untuk mencetak semua isi dari suatu direktori
    for root, dirs, files in os.walk(dir_path):
        st.write(f'Direktori: {root}')
        for file_name in files:
            st.write(f'  - {file_name}')



st.title('Dashboard - Safety Stock')

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False

def download_file_from_google_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

file_id = '1LM2YWeFHBxER3fqR0cY_OuP8x5FnZyeu'
dest_path = f'downloaded_file.zip'
download_file_from_google_drive(file_id, dest_path)
        
if 'df_cab' not in locals():
  with zipfile.ZipFile(f'downloaded_file.zip', 'r') as z:
    with z.open('df_month.csv') as f:
        df_month = pd.read_csv(f)
    with z.open('df_quarter.csv') as f:
        df_quarter = pd.read_csv(f)

list_bulan = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December']

quarter = st.selectbox("QUARTER:", ['Q1','Q2','Q3','Q4'], index=0, on_change=reset_button_state)
if quarter =='Q1':
    bulan = ['January','February','March']
if quarter =='Q2':
    bulan = ['April','May','June']
if quarter =='Q3':
    bulan = ['July','August','September']
else:
    bulan = ['October','November','December']

def format_number(x):
    if x==0:
        return ''
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x)
    return x

def highlight_indikator(val):
    if val == 'OK':
        color = 'background-color: green; color: green;'
    elif val == 'OVER':
        color = 'background-color: red; color: red;'
    elif val == 'LOWER':
        color = 'background-color: yellow; color: yellow;'
    else:
        color = ''
    return color
    
st.dataframe(df_quarter.style.format(lambda x: format_number(x)).applymap(highlight_indikator, subset=['INDIKATOR']), use_container_width=True, hide_index=True)

df_month['Month'] = pd.Categorical(df_month['Month'],categories=pd.to_datetime(df_month['Month'],format='%B').sort_values().dt.strftime('%B').unique())
df_month = df_month.sort_values(['Month','Nama Barang'])

df_3m = pd.concat([df_month[df_month['Month'].isin(bulan)].pivot(index='Nama Barang', columns=['Month'], values='AVG PICK UP').reset_index().merge(
    df_quarter[df_quarter['Quarter']==qr][['Nama Barang','AVG PICK UP']],how='left'),
    df_month[df_month['Month'].isin(bulan)].pivot(index='Nama Barang', columns=['Month'], values='AVG PEMBELIAN').reset_index().merge(
    df_quarter[df_quarter['Quarter']==qr][['Nama Barang','AVG PEMBELIAN','HARGA PEMBELIAN TERAKHIR','TOTAL']],how='left').drop(columns='Nama Barang'),
    df_month[df_month['Month'].isin(bulan)].pivot(index='Nama Barang', columns=['Month'], values='AVG SALDO AKHIR').reset_index().merge(
    df_quarter[df_quarter['Quarter']==qr][['Nama Barang','AVG SALDO AKHIR']],how='left').drop(columns='Nama Barang'),
    ], axis=1)

df_3m = df_3m.merge(df_quarter[['Nama Barang','Angka Standart','Indikator']],how='left')

st.dataframe(df_3m.style.format(lambda x: format_number(x)).applymap(highlight_indikator, subset=['INDIKATOR']), use_container_width=True, hide_index=True)
