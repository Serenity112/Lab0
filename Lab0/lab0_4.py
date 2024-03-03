import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def analyze_dataframe(df):
    print("Информация об индексах: ", df.index, "\n")
    print("Информация о типах данных: ", df.dtypes, "\n")
    print("Описательная статистика: ", df.describe(), "\n")
    print("Первые 5 строк для первых 19 столбцов: ", df[df.columns[:19]].head(5), "\n")

def yes_no_frames(df):
    df_yes = df.loc[df[df.columns[3]] == 'Yes']
    df_no = df.loc[df[df.columns[3]] == 'No']
    print("Yes")
    print(df_yes[df_yes.columns[:4]].head(10))
    print("No")
    print(df_no[df_no.columns[:4]].head(10))
    
def sort_df(df):
    sorted_df = df.sort_values(by=[df.columns[2], df.columns[3], 'Age'])
    print(sorted_df[sorted_df.columns[:6]].head(30))

def is_null_df(df):
    print("isna")
    print(df.isna())  
    print("dropna")
    print(df.dropna())

def make_gistogram(df):
    df_yes = df.loc[df[df.columns[3]] == 'Yes']
    df_no = df.loc[df[df.columns[3]] == 'No']
    plt.figure(figsize=(12, 6))

    # Гистограмма для Polyuria == 'Yes'
    plt.subplot(1, 2, 1)
    plt.hist(df_yes['Age'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Age Distribution (Polyuria == Yes)')
    plt.xlabel('Age')
    plt.ylabel('Count')

    # Гистограмма для Polyuria == 'No'
    plt.subplot(1, 2, 2)
    plt.hist(df_no['Age'], bins=10, color='salmon', edgecolor='black')
    plt.title('Age Distribution (Polyuria == No)')
    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def make_boxplot(df):
    yes_age = df[df['Polyuria'] == 'Yes']['Age']
    no_age = df[df['Polyuria'] == 'No']['Age']

    # boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([yes_age, no_age], labels=['Yes', 'No'])
    plt.xlabel('Polyuria')
    plt.ylabel('Age')
    plt.title('Boxplot распределения возраста')
    plt.grid(True)
    plt.show()

def make_scatter_matrix():

    data = pd.read_csv("csv1.csv")
    pd.plotting.scatter_matrix(data.iloc[:,[0,2,3]], c= data["class"].replace(["1","0"],["blue","red"]))
    plt.show()

table = pd.read_csv("csv0.csv")
#analyze_dataframe(table)
yes_no_frames(table)
#sort_df(table)
#is_null_df(table)
#make_gistogram(table)
#make_boxplot(table)
#make_scatter_matrix()
