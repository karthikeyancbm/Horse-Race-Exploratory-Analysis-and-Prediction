import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

with open("model.pkl","rb") as file:
    mdl = pickle.load(file)

age_list = [2, 3, 4, 5, 6, 7, 8]

fav_list = [0,1]

position_list = [0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341003, 1.791759469228055, 1.9459101490553132, 2.0794415416798357, 2.1972245773362196, 2.302585092994046, 2.3978952727983707, 2.4849066497880004, 2.5649493574615367, 2.6390573296152584, 2.70805020110221, 2.772588722239781, 2.833213344056216, 2.8903717578961645, 2.9444389791664403, 2.995732273553991, 3.044522437723423, 3.091042453358316]

post_L_list = [0.0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341003, 1.791759469228055, 1.9459101490553132, 2.0794415416798357, 2.1972245773362196, 2.302585092994046, 2.3978952727983707, 2.4849066497880004, 2.5649493574615367]

weight_list = [52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0]

res_win_list = [0, 1]

def race_pred(input_data):
    input_data_array = np.array(input_data)
    race_prediction = mdl.predict([input_data_array])
    return 'Won/Placed' if race_prediction[0] == 1 else 'Not_won/Not_placed'

df_1 = pd.read_csv(r"C:\Users\ASUS\Documents\GUVI ZEN CLASSES\MAINT BOOT\Horse Race Prediction\horse data\horses_2019.csv")
df_2 = pd.read_csv(r"C:\Users\ASUS\Documents\GUVI ZEN CLASSES\MAINT BOOT\Horse Race Prediction\race data\races_2019.csv")
df_3 = pd.read_csv(r"C:\Users\ASUS\Documents\GUVI ZEN CLASSES\MAINT BOOT\Horse Race Prediction\horse data\horses_2020.csv")
df_4 = pd.read_csv(r"C:\Users\ASUS\Documents\GUVI ZEN CLASSES\MAINT BOOT\Horse Race Prediction\race data\races_2020.csv")
df_eda = pd.concat([df_1,df_2,df_3,df_4],ignore_index = True)
cols = ['rid','saddle','positionL','dist','weightSt', 'weightLb', 'overWeight', 'outHandicap', 'headGear', 'RPR','TR', 'OR',
         'father', 'mother', 'gfather', 'runners', 'margin','res_place','course','time','title','band','condition','hurdles','prizes','winningTime',
         'prize','metric','class','ncond','ages']

df_eda.drop(columns=cols,inplace=True)

df_eda.drop(columns=['price','currency'],inplace=True)

df_eda['date'] = df_eda['date'].astype(str)

df_eda['date'] = df_eda['date'].str.strip()

def date_conversion(date):
    return date.split()[0]

df_eda['date'] = df_eda['date'].apply(lambda x: date_conversion(x))

df_eda['date'] = pd.to_datetime(df_eda['date'],errors='coerce')

df_eda['date'] = df_eda['date'].fillna(df_eda['date'].mode()[0])

df_eda['year'] = pd.to_datetime(df_eda['date'],errors='coerce').dt.year

df_eda['day'] = pd.to_datetime(df_eda['date'],errors='coerce').dt.day_name()

df_eda['month'] = pd.to_datetime(df_eda['date'],errors='coerce').dt.month_name()

df_eda.drop_duplicates(inplace=True)

def distance_conversion(distance):
    distance = str(distance)
    fractions = {'½':0.5,'¼':0.25,'¾':0.75}
    miles,furlong = 0,''
    if 'm' in distance:
        miles_part,furlong_part = distance.split('m')
        miles = int(miles_part)*8
        furlong = furlong_part.replace('f','')
    elif 'f' in distance:
        furlong = distance.replace('f','')
    for frac,value in fractions.items():
        furlong = furlong.replace(frac,str(value))
    return miles+float(furlong) if furlong else miles

df_eda['distance'] = df_eda['distance'].apply(lambda x: distance_conversion(x))

num_cols = df_eda.drop(columns=['res_win','isFav']).select_dtypes(include=float)

skew_cols = num_cols.drop(['age','position'],axis=1)

for col in num_cols:
    df_eda[col] = df_eda[col].fillna(df_eda[col].median())

df_eda['res_win'] = df_eda['res_win'].fillna(df_eda['res_win'].mode()[0])
df_eda['isFav'] = df_eda['isFav'].fillna(df_eda['isFav'].mode()[0])

for col in skew_cols:
    if df_eda[col].min() < 0:
        df_eda[col] = abs(df_eda[col])
    df_eda[col] = df_eda[col].apply(lambda x: np.log(x+1))

df_eda['age'] = df_eda['age'].astype(int)
df_eda['isFav'] = df_eda['isFav'].astype(int)
df_eda['res_win'] = df_eda['res_win'].astype(int)
df_eda['position'] = df_eda['position'].astype(int)

cat_cols = df_eda.drop(columns=['date','day','month']).select_dtypes(include=object)

for col in cat_cols:
    df_eda[col] = df_eda[col].fillna(df_eda[col].mode()[0])

for col in num_cols:
    q1 = df_eda[col].quantile(0.25)
    q3 = df_eda[col].quantile(0.75)
    iqr = q3 - q1
    lf = q1 - 1.5 * iqr
    uf = q3 + 1.5 * iqr
    df_eda[col] = df_eda[(df_eda[col]>lf) | (df_eda[col]<uf)][col]

df_eda = df_eda.reset_index(drop=True)

df_eda = df_eda[df_eda['decimalPrice']<0.23]

df_eda = df_eda.reset_index(drop=True)

df_eda = df_eda[df_eda['position']<15]

df_eda = df_eda.reset_index(drop=True)

df_eda = df_eda[df_eda['weight']>3.95]

df_eda = df_eda[df_eda['weight']<4.15]

df_eda = df_eda.reset_index(drop=True)

df_eda = df_eda[df_eda['age']<9]

df_eda = df_eda.reset_index(drop=True)

df_eda['distance'] = df_eda['distance'].fillna(df_eda['distance'].median())

df_eda.drop(columns=['distance'],inplace=True)

df_eda['weight'] = df_eda['weight'].astype(float)

df_eda['weight'] = df_eda['weight']*10

df_eda['weight'] = df_eda['weight'].apply(lambda x: f"{x:.2f}")

df_eda['weight'] = df_eda['weight'].astype(float)

df_eda['res_win'] = df_eda['res_win'].map({0:'Lost',1:"Win"})

# ## Total count of win and lost ##

def target_count(df_eda):
    return df_eda['res_win'].value_counts().reset_index()

def total_count(df_eda):
    with sns.color_palette("dark:#5A9_r"):
        fig,ax= plt.subplots(figsize=(8,6))
        sns.countplot(df_eda,x='res_win')
        ax.bar_label(container=ax.containers[0])
        ax.bar_label(container=ax.containers[0])
        ax.set_title('Win/Lost distribution')
        return fig
    
def total_count_pie(df_eda):
    fig,ax= plt.subplots(figsize=(10,8))
    figure = df_eda['res_win'].value_counts().plot(kind='pie',autopct ="%.2f")
    ax.set_title('Win/Lost distribution in percentage')
    return fig
    
## Year wise win/lost distribution ##

def year_analysis(df_eda):
    return df_eda[['year','res_win']].value_counts().reset_index()

def year_dist(df_eda):
    fig,ax = plt.subplots(figsize=(10,8))
    sns.countplot(df_eda,x='year',hue='res_win')
    plt.xticks(rotation=90)
    ax.set_title('Yearwise win/lost')
    return fig

## Monthwise win/lost distribution ##

def month_analysis(df_eda):
    return df_eda[['month','res_win']].value_counts().reset_index()

def month_dist(df_eda):
    fig,ax = plt.subplots(figsize=(10,8))
    sns.countplot(df_eda,x='month',hue='res_win')
    ax.bar_label(container=ax.containers[0])
    ax.bar_label(container=ax.containers[1])
    plt.xticks(rotation=90)
    ax.set_title('Monthwise win/lost')
    return fig

## Daywise win/lost distribution ##

def day_analysis(df_eda):
    return df_eda[['day','res_win']].value_counts().reset_index()

def day_dist(df_eda):
    fig,ax = plt.subplots(figsize=(10,8))
    sns.countplot(df_eda,x='day',hue='res_win')
    plt.xticks(rotation=90)
    ax.set_title('Daywise win/lost')
    return fig

## Age distribution for win/lost ##

def age_dist_df(df_eda):
    return df_eda[['res_win','age']]

def age_dist(df_eda):
    fig,ax =  plt.subplots(figsize=(12,10))
    sns.boxplot(x='age',y='res_win',data=df_eda)
    plt.xticks(rotation=90)
    ax.set_xlabel('Age')
    ax.set_ylabel('Win/Lost')
    ax.set_title("Age distribution")
    plt.tight_layout()
    return fig

## Average age ##

def avg_age(df_eda):
    avg_age = df_eda.groupby('res_win').agg({'age':'mean'}).reset_index()
    fig,ax  = plt.subplots(figsize=(12,10))
    ax.bar(avg_age['res_win'],avg_age['age'])
    ax.set_xlabel('Age')
    ax.set_title('Avearge Age')
    return fig

def avg_ag_df(df_eda):
    avg_age = df_eda.groupby('res_win').agg({'age':'mean'}).reset_index()
    return avg_age

## Minimum and Maximum age ##

def age_range(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    age_ran = df_eda['age'].agg(['min','max']).reset_index()
    ax.bar(age_ran['index'],age_ran['age'])
    ax.set_xlabel("Age")
    ax.set_title('Minimum & Maximum Age')
    return fig

def age_range_df(df_eda):
    return df_eda[['res_win','age']].agg(['min','max']).reset_index(drop=True)

## Winning frequency as per Age ##

def win_frq_age(df_eda):
    return df_eda.groupby('age')['res_win'].value_counts().reset_index()

def win_freq_age(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    sns.countplot(data=df_eda,x='age',hue='res_win')
    ax.set_xlabel('Age')
    ax.set_ylabel('win/lost count')
    ax.set_title('Winning frequency as per age')
    plt.tight_layout()
    return fig

## Weight distribution to win/lost  ##

def get_wt_dist(df_eda):
    wt_dist = df_eda[['weight','res_win']].value_counts().reset_index()
    wt_dist.drop(columns=['count'],inplace=True)
    return wt_dist


def weight_distribution(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    sns.boxplot(x='weight',y='res_win',data=df_eda)
    plt.xticks(rotation = 90)
    ax.set_xlabel('Weights')
    ax.set_ylabel('win/lost')
    ax.set_title('Weight distribution for win/lost')
    plt.tight_layout()
    return fig



## Weight distribution to win/lost percentage ##

def weight_dist_percent(df_eda):
    wt_count = df_eda.groupby('weight')['res_win'].value_counts(normalize=True).reset_index()
    return wt_count

def wt_to_win(df_eda):
    wt_count = df_eda.groupby('weight')['res_win'].value_counts(normalize=True).reset_index()
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(wt_count,x='weight',y='proportion',hue='res_win')
    ax.set_xlabel('Weights')
    ax.set_ylabel('Winning_percentage')
    ax.set_title('Weight distibution to the winning_percentage')
    return fig

## Winning_weight disribution ##


def win_wt_dist(df_eda):

    wt_count = df_eda[df_eda['res_win'] == 'Win']['weight'].value_counts().reset_index().sort_values(by='count',ascending=True)

    return wt_count


def win_weight_dist(df):
    wt_count = df[df['res_win'] == 'Win']['weight'].value_counts().reset_index().sort_values(by='count',ascending=True)
    fig,ax = plt.subplots(figsize=(12,10))
    sns.lineplot(wt_count,x='weight',y='count')
    ax.set_ylabel('Winning_weight_range')
    ax.set_title('Winning_weight distribution')
    return fig

## Lost weight distribution ##


def lost_wt_dist(df_eda):
    lost_wt = df_eda[df_eda['res_win'] == 'Lost']['weight'].value_counts().reset_index().sort_values(by='count',ascending=True)
    return lost_wt

def lost_weight_dist(df_eda):
    lost_wt = df_eda[df_eda['res_win'] == 'Lost']['weight'].value_counts().reset_index().sort_values(by='count',ascending=True)
    fig,ax=plt.subplots(figsize=(10,8))
    sns.lineplot(lost_wt,x='weight',y='count')
    plt.ylabel('Lossing weight range')
    plt.title('Lost weight distribution')
    return fig

## Classwise winning frequency ##

def class_win_freq(df_eda):
    return df_eda.groupby('rclass')['res_win'].value_counts().reset_index().sort_values(by='count',ascending=False)

def classwise_win(df_eda):
    fig,ax = plt.subplots(figsize=(10,8))
    sns.countplot(data=df_eda,x='rclass',hue='res_win')
    ax.set_xlabel('class_type')
    ax.set_ylabel('win/lost count')
    ax.set_title('Classwise winning frequency')
    plt.tight_layout()
    return fig

## Horses and their position ##

def horses_position(df_eda):
    post = df_eda.groupby('horseName')['position'].value_counts().reset_index().sort_values(by='count',ascending=False)
    return post

## Top 10 Horses which attained  no:1 position and its count ##

def top_10_horses(df_eda):
    top_10 = df_eda[df_eda['position'] == 1]['horseName'].value_counts().nlargest(10).reset_index()
    top_10.rename(columns={'horseName':'Horsenames','count':'Count'},inplace=True)
    return top_10

def top_10_plot(df_eda):
    top_10 = df_eda[df_eda['position'] == 1]['horseName'].value_counts().nlargest(10).reset_index()
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(top_10,x=top_10['horseName'],y=top_10['count'])
    ax.set_xlabel('Horse_names')
    ax.set_ylabel('position_count')
    ax.set_title('Top 10 Horses which attained no:1 position and its count')
    return fig
    
## Top 10 bottom Horses which attained last position ##

def bottom_10(df_eda):
    bottom_10 = df_eda[df_eda['position'] == 14]['horseName'].value_counts().nlargest(10).reset_index()
    bottom_10.rename(columns={'horseName':'Horsenames','count':'Count'},inplace=True)
    return bottom_10

def bottom_10_plot(df_eda):
    bottom_10 = df_eda[df_eda['position'] == 14]['horseName'].value_counts().nlargest(10).reset_index()
    fig,ax =  plt.subplots(figsize=(12,10))
    sns.barplot(bottom_10,x=bottom_10['horseName'],y=bottom_10['count'])
    plt.xticks(rotation=90)
    ax.set_xlabel('Horse_names')
    ax.set_ylabel('position_count')
    ax.set_title('Top bottom horses which attained last position and its count')
    return fig

## Countries won and lost ##

def country_win(df_eda):
    return df_eda.groupby('countryCode')['res_win'].value_counts().reset_index().sort_values(by='count',ascending=False)

def countriwise_result(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    sns.countplot(data=df_eda,x='countryCode',hue='res_win')
    plt.xticks(rotation=90)
    ax.set_title('Countries won and lost')
    return fig

## Unique countries for Win and Lost ##

def total_country_count(df_eda):
    win_country_count = df_eda[df_eda['res_win'] == 'Win']['countryCode'].nunique()
    win_1 = pd.Series(win_country_count,index=['GB'])
    lost_country_count = df_eda[df_eda['res_win'] == 'Lost']['countryCode'].nunique()
    lost_1 = pd.Series(lost_country_count,index=['Others'])
    total_country_count = pd.concat([win_1,lost_1],keys=['win', 'lost'],names=['category','country'])
    total_country_count.reset_index()
    total = pd.DataFrame(total_country_count.reset_index())
    total = total.rename(columns={0:'count'})
    return total


def unique_country(df_eda):
    win_country_count = df_eda[df_eda['res_win'] == 'Win']['countryCode'].value_counts().count()
    lost_country_count = df_eda[df_eda['res_win'] == 'Lost']['countryCode'].value_counts().count()
    categories = ['GB', 'Others']
    values = [win_country_count, lost_country_count]
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(x=categories,y= values,ax=ax,palette=['blue','orange'])
    ax.set_xlabel('Result')
    ax.set_ylabel('Number of Unique Countries')
    ax.set_title('Unique Countries for Win and Lost')
    ax.bar_label(ax.containers[0])
    ax.bar_label(ax.containers[1])
    return fig

## Fav and non-fav horses and their winning frequency ##

df_eda['isFav'] = df_eda['isFav'].map({0:'Not_fav',1:'Fav'})

def fav_horses_frq(df_eda):
    return df_eda.groupby('isFav')['res_win'].value_counts().reset_index()

def fav_horses_freq(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    sns.countplot(data=df_eda,x='isFav',hue='res_win')
    ax.set_xlabel('Category')
    ax.set_ylabel('Winning_count')
    ax.set_title('Fav and non-fav horses and their winning frequency')
    return fig

## Distribution of Decimal Odds (decimalPrice) ##

def decimal_price_dist(df_eda):
    return df_eda['decimalPrice'].value_counts().reset_index()

def decimal_dist(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    sns.histplot(df_eda['decimalPrice'], bins=50, kde=True)
    ax.set_title('Distribution of Decimal Odds (decimalPrice)')
    ax.set_xlabel('Decimal Price')
    ax.set_ylabel('Frequency')
    return fig

## Relationship between target and Odds ##

def get_relation(df_eda):
    return df_eda[['res_win','decimalPrice']]


def target_relt(df_eda):
    fig,ax = plt.subplots(figsize=(12,10))
    sns.boxplot(x='res_win', y='decimalPrice', data=df_eda)
    ax.set_title('Decimal Price vs Target')
    ax.set_xlabel('Target (Win/Lost)')
    ax.set_ylabel('Decimal Price')
    return fig

## Jockeys won race count ##

def get_jockeys_count(df_eda):
    jockey_count = df_eda[df_eda['res_win'] == 'Win']['jockeyName'].value_counts().reset_index().nlargest(10,'count')
    return jockey_count

def jockeys_won_count(df_eda):
    jockey_count = df_eda[df_eda['res_win'] == 'Win']['jockeyName'].value_counts().reset_index().nlargest(10,'count')
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(jockey_count,x='jockeyName',y = 'count',color='purple')
    plt.xticks(rotation = 90)
    ax.set_xlabel('Jockey_names')
    ax.set_ylabel('Winning_race_count')
    plt.tight_layout()
    return fig

## Jockeys Lost race count ##

def get_jockey_loss_count(df_eda):
    jockey_loss_count = df_eda[df_eda['res_win'] == 'Lost']['jockeyName'].value_counts().reset_index().nlargest(10,'count')
    return jockey_loss_count

def jockey_loss_count(df_eda):
    jockey_loss_count = df_eda[df_eda['res_win'] == 'Lost']['jockeyName'].value_counts().reset_index().nlargest(10,'count')
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(jockey_loss_count,x='jockeyName',y='count',color='purple')
    plt.xticks(rotation=90)
    ax.set_xlabel('Jockey_names')
    ax.set_ylabel('lost_race_count')
    ax.set_title('Jockeys Lost_race count')
    plt.tight_layout()
    return fig

## Bottom jockeys race count ##
def get_bottom_jockeys_count(df_eda):
    bottom_jockeys = df_eda.groupby('jockeyName')['res_win'].value_counts().reset_index().nsmallest(10,'count')
    return bottom_jockeys

def bottom_jockeys_count(df_eda):
    new = df_eda.groupby('jockeyName')['res_win'].value_counts().reset_index().nsmallest(10,'count')
    fig,ax=plt.subplots(figsize=(12,10)) 
    sns.countplot(data=new,x='jockeyName',hue='res_win')
    plt.xticks(rotation=90)
    ax.set_title('Bottom jockeys race count')
    plt.tight_layout()
    return fig                  

## Top 10 Trainers ##
def get_top_10_trainers(df_eda):
    top_trainers = df_eda[df_eda['res_win'] == 'Win']['trainerName'].value_counts().reset_index().nlargest(10,'count')
    return top_trainers

def top_10_trainers(df_eda):
    top_trainers = df_eda[df_eda['res_win'] == 'Win']['trainerName'].value_counts().reset_index().nlargest(10,'count')
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(top_trainers,x='trainerName',y='count',color='purple')
    plt.xticks(rotation=90)
    ax.set_xlabel('Trainer names')
    ax.set_title('Top 10 Trainers')
    plt.tight_layout()
    return fig

## Top 10 Lost Trainers ##

def get_lost_trainers(df_eda):
    lost_trainers = df_eda[df_eda['res_win'] == 'Lost']['trainerName'].value_counts().reset_index().nlargest(10,'count')
    return lost_trainers

def top_lost_trainers(df_eda):
    lost_trainers = df_eda[df_eda['res_win'] == 'Lost']['trainerName'].value_counts().reset_index().nlargest(10,'count')
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(lost_trainers,x='trainerName',y='count',color='purple')
    plt.xticks(rotation=90)
    ax.set_xlabel('Trainer Names')
    ax.set_title('Top 10 Loosing Trainers')
    plt.tight_layout()
    return fig

st.set_page_config(layout='wide')

title_text = '''<h1 style='font-size : 55px;text-align:center;color:purple;background-color:lightgrey;'>Horse Race Prediction</h1>'''
st.markdown(title_text,unsafe_allow_html=True)

with st.sidebar:

    select = option_menu('MAIN MENU',['HOME','ABOUT','EDA','PREDICTION'])

if select == 'HOME':
    st.write(" ")
    st.write(" ")   
    st.header(":blue[Horse race prediction - A Brief Overview]")

    with st.container(border=True):
      st.markdown('''<h6 style='color:#00ffff;font-size:21px'>People's interest in horse racing has skyrocketed along with its rapid expansion. 
                  Some experts and academics have studied the best practices for managing decisions and making predictions in horse racing. In the 
                  areas of categorization and prediction, applying the machine learning (ML) paradigms has demonstrated hopeful results. Betting on 
                  sports is big business, making accurate predictions in this field increasingly important. In addition, club executives are looking 
                  for classification models to better comprehend the game and develop winning plans. Research has shown that Machine Learning 
                  algorithms offer a good answer to the categorization and prediction problem in horse racing, where traditional prediction 
                  algorithms have failed. In this study, we present several ML approaches for predicting the outcome of horse races, including 
                  LogisticRegression(),AdaBoostClassifier(),RandomForestClassifier(),SVC(),DecisionTreeClassifier().
                  The AdaBoostClassifier approach produced more accurate predictions than any of the other models tested.''', unsafe_allow_html=True)
            
    st.header(":blue[Project Goals:]")

    st.subheader(":green[Primary goal:]")

    with st.container(border=True):
        st.markdown('''<h6 style='color:#00ffff;font-size:21px'>To predict the outcome of horse races (e.g., win or place)''',unsafe_allow_html=True)
    
    st.subheader(":red[Secondary goal:]")

    with st.container(border=True):
        st.markdown('''<h6 style='color:#00ffff;font-size:21px'>To identify significant features affecting race outcomes<br>
                    To explore the imbalanced nature of the dataset and develop techniques
                    to handle it.<br>To create a robust prediction model using historical data.''',unsafe_allow_html=True)

elif select == 'ABOUT':
    
    st.write(" ")
    st.write(" ")

    st.markdown('''<h6 style ='color:#ff1a66;font-size:31px'><br>Project Title : Horse Race Prediction''',unsafe_allow_html=True)


    st.markdown('''<h6 style ='color:#007acc;font-size:31px'><br>Domain : Sports ''',unsafe_allow_html=True)

    st.markdown('''<h6 style ='color:#1aff41;font-size:31px'><br>Take away Skills :''',unsafe_allow_html=True)

    st.markdown('''<h6 style ='color:#00ffff;font-size:31px'>Python Scripting<br>Data Cleaning & pre-processing<br>EDA<br>Visualisation<br>Model Building<br>Model Deployment in 
                    Streamlit''',unsafe_allow_html=True)
    
elif select == 'EDA':
    
    st.write(" ")
    st.write(" ")

    option = st.selectbox("Select the Queries to ba analysed : ",
                          
    ("1.Total count of win and lost",
    "2.Year wise win/lost distribution",
    "3.Monthwise win/lost distribution",
    "4.Daywise win/lost distribution",
    "5.Age distribution for win/lost",
    "6.Average age",
    "7.Minimum and Maximum age",
    "8.Winning frequency as per Age",
    "9.Weight distribution to win/lost",
    "10.Weight distribution to win/lost percentage",
    "11.Winning_weight disribution",
    "12.Lost weight distribution",
    "13.Classwise winning frequency",
    "14.Top 10 Horses which attained  no:1 position and its count",
    "15.Top 10 bottom Horses which attained last position",
    "16.Countries won and lost",
    "17.Unique countries for Win and Lost",
    "18.Fav and non-fav horses and their winning frequency",
    "19.Distribution of Decimal Odds (decimalPrice)",
    "20.Relationship between target and Odds",
    "21.Jockeys won race count",
    "22.Jockeys Lost race count",
    "23.Bottom jockeys race count",
    "24.Top 10 Trainers",
    "25.Top 10 Lost Trainers",
     ),
    index=None,
    placeholder="Select the Query...",
    )

    st.write('You selected:', option)

    if option == "1.Total count of win and lost":

        col1,col2 = st.columns(2)

        with col1:

            target_count_frame = target_count(df_eda)
            st.dataframe(target_count_frame,width=600,height=100)

            fig_1 = total_count_pie(df_eda)
            st.pyplot(fig_1)

        with col2:

            fig = total_count(df_eda)
            st.pyplot(fig)

            
    if option == "2.Year wise win/lost distribution":

        col1,col2 = st.columns(2)

        with col1:

            year_df = year_analysis(df_eda)
            st.dataframe(year_df,width=600,height=500)

        with col2:

            fig = year_dist(df_eda)
            st.pyplot(fig)

    if option == "3.Monthwise win/lost distribution":

        col1,col2 =st.columns(2)

        with col1:
            month_df = month_analysis(df_eda)
            st.dataframe(month_df,width=600,height=510)
        
        with col2:
            fig = month_dist(df_eda)
            st.pyplot(fig)

    if option == "4.Daywise win/lost distribution":

        col1,col2 = st.columns(2)

        with col1:
            day_df = day_analysis(df_eda)
            st.dataframe(day_df,width=600,height=350)

        fig = day_dist(df_eda)
        st.pyplot(fig)

    if option == "5.Age distribution for win/lost":

        col1,col2 = st.columns(2)

        with col1:
            age_distribution_df = age_dist_df(df_eda)
            st.dataframe(age_distribution_df,width=600,height=500)

        with col2:

            fig = age_dist(df_eda)
            st.pyplot(fig)

    if option == "6.Average age":

        col1,col2 = st.columns(2)

        with col1:
            avg_age_datframe = avg_ag_df(df_eda)
            st.dataframe(avg_age_datframe,width=600,height=100)

        with col2:

            fig = avg_age(df_eda)
            st.pyplot(fig)

    if option == "7.Minimum and Maximum age":

        col1,col2 = st.columns(2)
        with col1:
            age_range_datframe = age_range_df(df_eda)
            st.dataframe(age_range_datframe,width=600,height=100)

        with col2:

            fig = age_range(df_eda)
            st.pyplot(fig)

    
    if option == "8.Winning frequency as per Age":

        col1,col2 = st.columns(2)

        with col1:

            win_frq_df = win_frq_age(df_eda)
            st.dataframe(win_frq_df,width=600,height=510)

        with col2:

            fig = win_freq_age(df_eda)
            st.pyplot(fig)

    if option == "9.Weight distribution to win/lost":

        col1,col2 =st.columns(2)

        with col1:
            weightt_dist_df = get_wt_dist(df_eda)
            st.dataframe(weightt_dist_df,width=600,height=510)

        with col2:

            fig = weight_distribution(df_eda)
            st.pyplot(fig)

    if option == "10.Weight distribution to win/lost percentage":

        col1,col2 = st.columns(2)

        with col1:
            wgt_dist_percent_df = weight_dist_percent(df_eda)
            st.dataframe(wgt_dist_percent_df,width=600,height=510)
        with col2:

            fig = wt_to_win(df_eda)
            st.pyplot(fig)

    if option == "11.Winning_weight disribution":

        col1,col2 =st.columns(2)

        with col1:

            win_wt_dist_df = win_wt_dist(df_eda)
            st.dataframe(win_wt_dist_df,width=600,height=510)
        with col2:

            fig = win_weight_dist(df_eda)
            st.pyplot(fig)

    if option == "12.Lost weight distribution":

        col1,col2 =st.columns(2)

        with col1:

            lost_wt_df = lost_wt_dist(df_eda)
            st.dataframe(lost_wt_df,width=600,height=510)
        
        with col2:

            fig = lost_weight_dist(df_eda)
            st.pyplot(fig)

    if option == "13.Classwise winning frequency":

        col1,col2 =st.columns(2)

        with col1:
            classwise_win_df = class_win_freq(df_eda)
            st.dataframe(classwise_win_df,width=600,height=330)

        with col2:

            fig = classwise_win(df_eda)
            st.pyplot(fig)

    if option == "14.Top 10 Horses which attained  no:1 position and its count":

        col1,col2 =st.columns(2)

        with col1:
            top_horse_df = top_10_horses(df_eda)
            st.dataframe(top_horse_df,width=600,height=510)

        with col2:

            fig = top_10_plot(df_eda)
            st.pyplot(fig)

    if option == "15.Top 10 bottom Horses which attained last position":

        col1,col2 =st.columns(2)

        with col1:

            bottom_horse_df = bottom_10(df_eda)
            st.dataframe(bottom_horse_df,width=600,height=400)

        with col2:
            fig = bottom_10_plot(df_eda)
            st.pyplot(fig)

    if option == "16.Countries won and lost":

        col1,col2=st.columns(2)

        with col1:

            country_win_df = country_win(df_eda)
            st.dataframe(country_win_df,width=600,height=500)
        with col2:
            fig = countriwise_result(df_eda)
            st.pyplot(fig)

    if option == "17.Unique countries for Win and Lost":

        col1,col2 =st.columns(2)

        with col1:
            unique_country_df = total_country_count(df_eda)
            st.dataframe(unique_country_df,width=600,height=100)
        with col2:
            fig = unique_country(df_eda)
            st.pyplot(fig)

    if option == "18.Fav and non-fav horses and their winning frequency":

        col1,col2 =st.columns(2)

        with col1:
            fav_horse_df = fav_horses_frq(df_eda)
            st.dataframe(fav_horse_df,width=600,height=200)
        with col2:
            fig = fav_horses_freq(df_eda)
            st.pyplot(fig)

    if option == "19.Distribution of Decimal Odds (decimalPrice)":

        col1,col2 = st.columns(2)

        with col1:
            decimal_price_df = decimal_price_dist(df_eda)
            st.dataframe(decimal_price_df,width=600,height=510)

        with col2:
            fig = decimal_dist(df_eda)
            st.pyplot(fig)

    if option == "20.Relationship between target and Odds":

        col1,col2 =st.columns(2)

        with col1:
            relations_df = get_relation(df_eda)
            st.dataframe(relations_df,width=600,height=510)
        with col2:
            fig = target_relt(df_eda)
            st.pyplot(fig)

    if option == "21.Jockeys won race count":

        col1,col2 = st.columns(2)

        with col1:
            jockeys_count_df = get_jockeys_count(df_eda)
            st.dataframe(jockeys_count_df,width=600,height=450)

        with col2:

            fig = jockeys_won_count(df_eda)
            st.pyplot(fig)

    if option == "22.Jockeys Lost race count":

        col1,col2 = st.columns(2)
        with col1:
            jockey_loss_count_df = get_jockey_loss_count(df_eda)
            st.dataframe(jockey_loss_count_df,width=600,height=420)
        with col2:
            fig = jockey_loss_count(df_eda)
            st.pyplot(fig)

    if option == "23.Bottom jockeys race count":

        col1,col2 = st.columns(2)

        with col1:
            bottom_jockeys_count_df= get_bottom_jockeys_count(df_eda)
            st.dataframe(bottom_jockeys_count_df,width=600,height=410)
        with col2:
            fig = bottom_jockeys_count(df_eda)
            st.pyplot(fig)

    if option == "24.Top 10 Trainers":

        col1,col2 = st.columns(2)

        with col1:
            top_10_trainers_df = get_top_10_trainers(df_eda)
            st.dataframe(top_10_trainers_df,width=600,height=410)

        with col2:
            fig = top_10_trainers(df_eda)
            st.pyplot(fig)

    if option == "25.Top 10 Lost Trainers":

        col1,col2 = st.columns(2)

        with col1:
            lost_trainers_df = get_lost_trainers(df_eda)
            st.dataframe(lost_trainers_df,width=600,height=410)
        with col2:
            fig = top_lost_trainers(df_eda)
            st.pyplot(fig)

elif select == 'PREDICTION':
    
    title_text = '''<h1 style ='font-size: 30px;text-align: center;color:#00ff80;'>To Predict the Race outcome,please provide the following 
                information</h1>'''
    st.markdown(title_text,unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    with col1:

        age = st.selectbox("Age",age_list,index=None)

        decimal_price = st.number_input('Decimal_price',min_value=0.00,max_value=0.26)

        isFav = st.selectbox('Favorite',fav_list,index=None)

        position = st.selectbox('Position',position_list,index=None)

        positionL = st.selectbox('PositionL',post_L_list,index=None)

        dist = st.number_input('Dist',min_value=0.00,max_value=6.99)

    with col2:

        rpr = st.number_input('RPR',min_value=3.40,max_value=4.96)

        tr = st.number_input('TR',min_value=0.00,max_value=4.83)

        Or = st.number_input('OR',min_value=0.0,max_value=125.0)

        margin = st.number_input('Margin',min_value=1.08,max_value=1.31)

        weight = st.selectbox('Weight',weight_list,index=None)

        res_win = st.selectbox('res_win',res_win_list,index=None)

    with col1:
        st.write(" ")
        st.write(" ")

    if st.button(':violet[Predict]',use_container_width=True):

        prediction = race_pred([age,decimal_price,isFav,position,positionL,dist,rpr,tr,Or,margin,weight,res_win])

        st.subheader((f":green[Race outcome :] {prediction}"))