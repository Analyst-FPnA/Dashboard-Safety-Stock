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
import numpy as np

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



st.title('Dashboard - Safety Stock & Movement Equipment')
st.markdown('#### Stock Over Standard')

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Fungsi untuk mereset state button
def reset_button_state():
    st.session_state.button_clicked = False

def download_file_from_google_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

        
file_id = '1kFrECcZmJ1SgvexIvcMY_tsg45nDxobH'
dest_path = f'downloaded_file.zip'
download_file_from_google_drive(file_id, dest_path)
if not os.path.exists('all_4208.csv'):
    with zipfile.ZipFile(f'downloaded_file.zip', 'r') as z:
          df_4208 = []
          df_4205 = []
          df_9901 = []
          for file in z.namelist():
            if file.startswith('42.08'):  
              # Loop untuk membaca setiap file di dalam ZIP
                  with z.open(file) as f:
                      # Membaca setiap file Excel ke dalam DataFrame
                      df = pd.read_excel(f)
                      df_4208.append(df)
            if file.startswith('42.05'):  
              # Loop untuk membaca setiap file di dalam ZIP
                  with z.open(file) as f:
                      # Membaca setiap file Excel ke dalam DataFrame
                      df = pd.read_excel(f)
                      df_4205.append(df)
            if file.startswith('99.01'):  
              # Loop untuk membaca setiap file di dalam ZIP
                  with z.open(file) as f:
                      # Membaca setiap file Excel ke dalam DataFrame
                      df = pd.read_csv(f)
                      df_9901.append(df)
                    
          # Menggabungkan semua DataFrame
          pd.concat(df_4208, ignore_index=True).to_csv('all_4208.csv',index=False)
          pd.concat(df_4205, ignore_index=True).to_csv('all_4205.csv',index=False)
          pd.concat(df_9901, ignore_index=True).to_csv('all_9901.csv',index=False)
        
if 'df_cab' not in locals():
  with zipfile.ZipFile(f'downloaded_file.zip', 'r') as z:
    with z.open('daftar_gudang.csv') as f:
        df_cab = pd.read_csv(f)
    with z.open('Stocklevel.xlsx') as f:
        df_level = pd.read_excel(f)

df = pd.read_csv('all_4208.csv')
df_it = pd.read_csv('all_4205.csv')
df['Cabang'] = df['Cabang'].replace({'System)':'Transit (AOL System)'})
df['Cabang'] = df['Cabang'].str.extract(r'\(([^()]*)\)[^()]*$')
df = df.merge(df_cab[['Cabang','Nama Cabang']],how='left')
df['Bulan'] = pd.to_datetime(df['Tanggal'],format='%d/%m/%Y',errors='raise').dt.month_name()

list_bulan = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December']

df['Bulan'] = df['Bulan'].bfill()
with zipfile.ZipFile(f'downloaded_file.zip', 'r') as z:
    with z.open(f'4201_September.xlsx') as f:
        df_4201 = pd.read_excel(f,header=4).loc[1:,['Nama Barang','Total Nama Gudang']]

df_9901 = pd.read_csv('all_9901.csv')
df_9901['Tanggal']  = pd.to_datetime(df_9901['Tanggal'])
df_9901['#Purch.@Price'] = df_9901['#Purch.@Price'].astype(float)

dfs = []
for b in df['Bulan'].unique():
    if b=='January':
        df_saldo = df[~(df['Bulan'].isin(list_bulan[list_bulan.index(b)+1:]))&((df['Nama Cabang'].str.startswith('H00')) | (df['Nama Cabang'].str.startswith('2')) | (df['Nama Cabang'].str.startswith('5')))]
        df_saldo = df_saldo[df_saldo['Deskripsi'].str.contains('Saldo')].groupby(['Nama Barang'])[['Masuk']].sum().reset_index().rename(columns={'Masuk':'Saldo Akhir'})
        df_saldo['Bulan'] = b
        dfs.append(df_saldo[['Bulan','Nama Barang','Saldo Akhir']])
    else:
        df_saldo = df[~(df['Bulan'].isin(list_bulan[list_bulan.index(b)+1:]))&((df['Nama Cabang'].str.startswith('H00')) | (df['Nama Cabang'].str.startswith('2')) | (df['Nama Cabang'].str.startswith('5')))]
        df_saldo = df_saldo.groupby(['Nama Barang'])[['Masuk']].sum().reset_index().merge(df_saldo.groupby(['Nama Barang'])[['Keluar']].sum().reset_index(),how='outer')
        df_saldo['Saldo Akhir'] =  df_saldo['Masuk'] - df_saldo['Keluar']
        df_saldo['Bulan'] = b
        dfs.append(df_saldo[['Bulan','Nama Barang','Saldo Akhir']])
df_over = pd.concat(dfs,ignore_index=True)

dfs=[]
for b in ['July','August','September']:
    df_tab = df[(df['Bulan'].isin(list_bulan[list_bulan.index(b)-6:list_bulan.index(b)])) & ((df['Cabang']=='IT') |(df['Nama Cabang'].str.startswith('2')) | (df['Nama Cabang'].str.startswith('5')))]
    df_saldo = df_saldo.merge(df_tab[(df_tab['Nomor #'].str.contains('RI.')) | (df_tab['Nomor #'].str.contains('PI.'))].groupby('Nama Barang')[['Masuk']].sum().reset_index(), how='left')
    df_kirim = df_tab[(df_tab['Keluar']!=0) & (df_tab['Nomor #'].str.contains('IT'))]
    df_kirim = df_kirim.merge(df_it.drop_duplicates(subset=['Nomor #Kirim','Nama Barang']), how='left',left_on=['Nomor #','Nama Barang'], right_on=['Nomor #Kirim','Nama Barang'])
    df_std = df_kirim[(df_kirim['Gudang #Terima'].str.contains('|'.join(df_cab[(df_cab['Nama Cabang'].str.startswith('1')) | (df_cab['Nama Cabang'].str.startswith('9'))]['Nama Cabang'].str[:6].values)))]
    df_std = df_std.groupby(['Nama Barang','Bulan'])[['Keluar']].sum().reset_index()
    df_std = df_std[df_std['Keluar']>0]
    df_std = df_std.groupby(['Nama Barang'])[['Keluar']].mean().astype('int').reset_index()
    df_std['Keluar'] = ((df_std['Keluar'].fillna(0)*0.1)+df_std['Keluar'].fillna(0)).astype(int)
    df_std['Month'] = b
    df_std = df_std.merge(df_4201[['Kode Barang','Nama Barang']].drop_duplicates(),how='left')
    df_std['Kode Barang'] = df_std['Kode Barang'].astype('int').astype('str')
    data = df_9901[df_9901['Tanggal']<pd.to_datetime(f'{list_bulan[list_bulan.index(b)+1]} 2024',format='%B %Y')].sort_values(['Tanggal','Kode #','#Purch.@Price'],ascending=False).drop_duplicates(subset=['Kode #'])
    data['Kode #'] = data['Kode #'].astype('int').astype('str')
    df_std = df_std.merge(data[['Kode #','#Purch.@Price']],how='left',left_on=['Kode Barang'],right_on=['Kode #'])
    dfs.append(df_std)

#df_over = df_over.merge(df_std,how='left')

#df_over = df_over.pivot(index='Nama Barang',columns='Bulan',values='Overstock').reset_index()#.fillna(0)
#total = pd.DataFrame([['TOTAL BARANG OVERSTOK']+df_over.iloc[:,1:].count(axis=0).values.tolist()],columns=df_over.columns)

def format_number(x):
    if x==0:
        return ''
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x)
    return x

df_over = df_over.merge(pd.concat(dfs,ignore_index=True).rename(columns={'Month':'Bulan'}))
df_over['Kuantitas'] = df_over['Saldo Akhir'] - df_over['Keluar']
df_over = df_over[df_over['Kuantitas']>0]
df_over['Nominal'] = df_over['Kuantitas']*df_over['#Purch.@Price']
agg =st.selectbox("KUANTITAS/NOMINAL:", ['Kuantitas','Nominal'], index=0, on_change=reset_button_state)
df_over['Bulan'] = pd.Categorical(df_over['Bulan'],categories=list_bulan)
df_over = df_over.pivot(index='Nama Barang',columns='Bulan',values=agg).reset_index()
total = pd.DataFrame([['TOTAL BARANG OVERSTOK']+df_over.iloc[:,1:].count(axis=0).values.tolist()],columns=df_over.columns)
    
st.dataframe(total.style.background_gradient(cmap='Reds', axis=1, subset=total.columns[1:]), use_container_width=True, hide_index=True)   
df_over = df_over.fillna(0).style.format(lambda x: format_number(x)).background_gradient(cmap='Reds', axis=1, subset=df_over.columns[1:])
st.dataframe(df_over, use_container_width=True, hide_index=True)


st.markdown('####')
bulan =st.selectbox("BULAN:", list_bulan, index=8, on_change=reset_button_state)

df_tab = df[(df['Bulan'].isin(list_bulan[list_bulan.index(bulan)-6:list_bulan.index(bulan)])) & ((df['Cabang']=='IT') |(df['Nama Cabang'].str.startswith('2')) | (df['Nama Cabang'].str.startswith('5')))]
df_saldo = df_saldo.merge(df_tab[(df_tab['Nomor #'].str.contains('RI.')) | (df_tab['Nomor #'].str.contains('PI.'))].groupby('Nama Barang')[['Masuk']].sum().reset_index().rename(columns={'Masuk':f'Pembelian {bulan}'}), how='left')
df_kirim = df_tab[(df_tab['Keluar']!=0) & (df_tab['Nomor #'].str.contains('IT'))]
df_kirim = df_kirim.merge(df_it.drop_duplicates(subset=['Nomor #Kirim','Nama Barang']), how='left',left_on=['Nomor #','Nama Barang'], right_on=['Nomor #Kirim','Nama Barang'])
df_std = df_kirim[(df_kirim['Gudang #Terima'].str.contains('|'.join(df_cab[(df_cab['Nama Cabang'].str.startswith('1')) | (df_cab['Nama Cabang'].str.startswith('9'))]['Nama Cabang'].str[:6].values)))]
df_std = df_std.groupby(['Nama Barang','Bulan'])[['Keluar']].sum().reset_index()
df_std = df_std[df_std['Keluar']>0]
df_std = df_std.groupby('Nama Barang')[['Keluar']].mean().astype('int').reset_index()

df_saldo = df[~(df['Bulan'].isin(list_bulan[list_bulan.index(bulan):])) & ((df['Nama Cabang'].str.startswith('H00')) |(df['Nama Cabang'].str.startswith('2')) | (df['Nama Cabang'].str.startswith('5')))]
df_saldo = df_saldo.groupby('Nama Barang')[['Masuk']].sum().reset_index().merge(df_saldo.groupby('Nama Barang')[['Keluar']].sum().reset_index(),how='outer')
df_saldo[f'SO Awal Bulan {bulan}'] =  (df_saldo['Masuk'] - df_saldo['Keluar']).astype('int')
df_saldo[f'SO Awal Bulan {bulan}'] = df_saldo[f'SO Awal Bulan {bulan}'].astype('int')
df_tab = df[(df['Bulan'] == bulan) & ((df['Nama Cabang'].str.startswith('H00')) | (df['Nama Cabang'].str.startswith('2')) | (df['Nama Cabang'].str.startswith('5')))]
df_saldo = df_saldo.merge(df_tab[df_tab['Deskripsi'].str.contains('Penerimaan')].groupby('Nama Barang')[['Masuk']].sum().reset_index().rename(columns={'Masuk':f'Pembelian {bulan}'}), how='left')
df_kirim = df_tab[(df_tab['Keluar']!=0) & (df_tab['Nomor #'].str.contains('IT'))]
df_kirim = df_kirim.merge(df_it.drop_duplicates(subset=['Nomor #Kirim','Nama Barang']), how='left',left_on=['Nomor #','Nama Barang'], right_on=['Nomor #Kirim','Nama Barang'])
#st.dataframe(df_kirim)
df_kirim = df_kirim[(df_kirim['Gudang #Terima'].str.contains('|'.join(df_cab[(df_cab['Nama Cabang'].str.startswith('1')) | (df_cab['Nama Cabang'].str.startswith('9'))]['Cabang']),case=False))]

df_saldo = df_saldo.merge(df_kirim.groupby('Nama Barang')[['Keluar']].sum().reset_index().rename(columns={'Keluar':f'Pickup Resto {bulan}'}),how='left')

with zipfile.ZipFile(f'downloaded_file.zip', 'r') as z:
    with z.open(f'4201_{bulan}.xlsx') as f:
        df_4201 = pd.read_excel(f,header=4).loc[1:,['Nama Barang','Total Nama Gudang']]


df_saldo = df_saldo.fillna(0)
df_saldo[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}'] = (df_saldo[f'SO Awal Bulan {bulan}'] + df_saldo[f'Pembelian {bulan}'] - df_saldo[f'Pickup Resto {bulan}']).astype(int)
df_saldo = df_saldo.drop(columns=['Masuk','Keluar'])
df_saldo.iloc[:,1:] = df_saldo.iloc[:,1:].astype(int)
df_saldo = df_saldo.merge(df_4201, how='left').rename(columns={'Total Nama Gudang':f'SO 42.01 {bulan}'})
df_level = df_level.rename(columns={'Nama Barang Barang & Jasa':'Nama Barang','Level Stock':'Angka Standart'})[['Nama Barang','Angka Standart']]
df_saldo = df_level.merge(df_saldo,how='left').merge(df_std,how='left')
df_saldo['Control'] = df_saldo[f'SO 42.01 {bulan}'] - df_saldo[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}']
def indikator(row):
    if row[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}']!=np.nan:
        markup = np.round((row['Keluar']*(10/100))+row['Keluar'])
        bm = markup + np.round(markup*(5/100))
        ba = markup - np.round(markup*(5/100))
        if (row[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}'] >= bm) & (row[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}'] <= ba):
            return 'Hijau'
        if (row[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}'] > ba):
            return 'Merah'
        if (row[f'SO Awal Bulan {list_bulan[list_bulan.index(bulan)+1]}'] < bm):
            return 'Kuning'
        
df_saldo['Indikator'] = df_saldo.apply(lambda row: indikator(row), axis=1)
def highlight_indikator(val):
    if val == 'Hijau':
        color = 'background-color: green; color: green;'
    elif val == 'Merah':
        color = 'background-color: red; color: red;'
    elif val == 'Kuning':
        color = 'background-color: yellow; color: yellow;'
    else:
        color = ''
    return color
df_saldo[f'Pembelian {bulan}'] = df_saldo[f'Pembelian {bulan}'].fillna(0).astype(int)   
df_saldo[f'Pickup Resto {bulan}'] = df_saldo[f'Pickup Resto {bulan}'].fillna(0).astype(int)

def format_number(x):
    if isinstance(x, (int, float)):
        return "{:,.0f}".format(x)
    return x
st.dataframe(df_saldo.rename(columns={'Keluar':f'Avg Pickup Resto {list_bulan[list_bulan.index(bulan)-6][:3]} - {list_bulan[list_bulan.index(bulan)-1][:3]}'}).iloc[:,[0,1,2,3,4,7,5,6,8,9]].style.format(lambda x: format_number(x)).applymap(highlight_indikator, subset=['Indikator']), use_container_width=True, hide_index=True)

barang = st.selectbox("NAMA BARANG:", df_level['Nama Barang'].values.tolist(), index=0, on_change=reset_button_state)
#barang = df_saldo['Nama Barang'].values[0]

df_it['Bulan'] = pd.to_datetime(df_it['Tanggal #Terima'], format='%d %b %Y').dt.month_name()
st.dataframe(df_it[(df_it['Bulan']==bulan)&((df_it['Gudang #Kirim'].str.startswith('2')) | (df_it['Gudang #Kirim'].str.startswith('5')))
      & ((df_it['Gudang #Terima'].str.startswith('1')) | (df_it['Gudang #Terima'].str.startswith('9')))
      & (df_it['Nama Barang']==barang)][['Nomor #Terima','Gudang #Terima','Tanggal #Terima','#Sat. Terkecil','#Qty. Terkecil']], use_container_width=True, hide_index=True)
