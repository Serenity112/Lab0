import csv
from prettytable import PrettyTable

def task1(data):
    sum_num = 0
    for i in range(1, len(data)):
        sum_num += int(data[i][0])
    avg = sum_num / (len(data))
    print("Средний возраст:")
    print(avg)

def task2(data):
    male_avg_age = 0
    male_count = 0
    female_avg_age = 0
    female_count = 0
    for i in range(1, len(data)):
        if data[i][1] == "Male":
            male_avg_age += int(data[i][0])
            male_count += 1
        else:
            female_avg_age += int(data[i][0])
            female_count += 1
    male_avg_age /= male_count
    female_avg_age /= female_count
    print("Средний возраст male:")
    print(male_avg_age)
    print("Средний возраст female:")
    print(female_avg_age)


def task3(data):
    ob_and_diab = 0
    healthy = 0
    obesities = 0
    diabets = 0
    for i in range(1, len(data)):
        if (data[i][-1] == "Positive") and (data[i][-2] == "Yes"):
            ob_and_diab += 1
        if (data[i][-1] == "Negative") and (data[i][-2] == "No"):
            healthy += 1
        if (data[i][-1] == "Positive") and (data[i][-2] == "No"):
            obesities += 1
        if (data[i][-1] == "Negative") and (data[i][-2] == "Yes"):
            diabets += 1
    table = PrettyTable()
    table.field_names = ["", "Diabetes +", "Diabetes -"]
    table.add_row(["Obesity +", ob_and_diab, obesities])
    table.add_row(["Obesity -", diabets, healthy])
    
    print(table)

with open('csv0.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

task1(data)
task2(data)
task3(data)




