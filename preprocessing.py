import csv
import numpy as np
import sklearn

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

data_patients = []
training_patients = []
validation_patients = []
data_events = []

def collectData():
    with open('all/Enrollment Data (collated final).csv', 'r', encoding='utf8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        csv_heading = next(spamreader)
        for row in spamreader:
            new_row = {}
            for i in range(len(csv_heading)):
                new_row[csv_heading[i]] = row[i]
            data_patients.append(new_row)

    with open('all/train.csv', 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        csv_heading = next(reader)
        for row in reader:
            new_row = {}
            for i in range(len(csv_heading)):
                new_row[csv_heading[i]] = row[i]
            training_patients.append(new_row)

    with open('all/test.csv', 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        csv_heading = next(reader)
        for row in reader:
            new_row = {}
            for i in range(len(csv_heading)):
                new_row[csv_heading[i]] = row[i]
            validation_patients.append(new_row)

    with open('all/Patient Engagement Events Data (collated final).csv', 'r', encoding='utf8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        csv_heading = next(spamreader)
        for row in spamreader:
            new_row = {}
            for i in range(len(csv_heading)):
                if csv_heading[i] == 'Event_Date':
                    yr = row[i].split('/')[2]
                    new_row[csv_heading[i]] = datetime.strptime(row[i], "%m/%d/%Y").date() if len(
                        yr) > 2 else datetime.strptime(row[i], "%m/%d/%y").date() + relativedelta(years=2000)
                else:
                    new_row[csv_heading[i]] = row[i]
            data_events.append(new_row)

def getInfo(input_X, patients, dates, birth, registration, scale):
    norm = np.zeros((len(input_X),4))
    for i in range(len(input_X)):
        one_patient_data = list(filter(lambda x: x['Patient Id'] == patients[i]['Patient Id'] and patients[i]['Hospital Id'] == x['\ufeffHospital Id'], data_events))
        appointment_date = dates[i]
        registration_date = registration[i]
        if birth[i]:
            age = appointment_date - birth[i]
            age = age.days/365
        else:
            age = 0
        dayof_before = appointment_date
        regist = appointment_date - registration_date
        day_14_before = appointment_date - timedelta(days=14)
        # get begining and after events we're looking at
        relevant_events = list(filter(lambda x: x['Event_Date'] < appointment_date and x['Event_Date'] > registration_date, one_patient_data))
        timeframe_0_14 = list(filter(lambda x: x['Event_Date'] < dayof_before and x['Event_Date'] > day_14_before, relevant_events))
        rescheduled = cancelled = copilot = unsubscribed = triple_opt_int = 0
        ui_click = node_viewed = user_compl = [0, 0]
        for event in timeframe_0_14:
            if event['Event_Name'] == 'User_clicked_UI_button':
                ui_click[0] += 1
            elif event['Event_Name'] == 'Node_Viewed':
                node_viewed[0] += 1
            elif event['Event_Name'] == 'User_completed_module':
                user_compl[0] += 1
        for event in relevant_events:
            if event['Event_Name'] == 'Patient_Rescheduled_By_Tenant':
                rescheduled = 1
            elif event['Event_Name'] == 'Patient_Cancelled_By_Tenant':
                cancelled = 1
            elif event['Event_Name'] == 'Added_Copilot':
                copilot = 1
            elif event['Event_Name'] == 'User_unsubscribed':
                unsubscribed = 1
        if type == "norm":
            norm[i][0] = ui_click[0]
            norm[i][1] = node_viewed[0]
            norm[i][2] = user_compl[0]
            norm[i][3] = age
        else:
            input_X[i][2] = ui_click[0]
            input_X[i][3] = node_viewed[0]
            input_X[i][4] = user_compl[0]
            input_X[i][9] = age
        input_X[i][5] = rescheduled
        input_X[i][6] = cancelled
        input_X[i][7] = copilot
        input_X[i][8] = unsubscribed
        input_X[i][10] = regist.days
    if scale == "norm":
        norm = sklearn.preprocessing.normalize(norm)
        input_X.T[2] = norm.T[0]
        input_X.T[3] = norm.T[1]
        input_X.T[4] = norm.T[2]
        input_X.T[9] = norm.T[3]
    elif scale == "MinMax":
        scaler = MinMaxScaler()
        scaler.fit(input_X)
        input_X = scaler.transform(input_X)
    return input_X

def getTrainingData(scale=None):
    Y = np.zeros((len(training_patients),1))
    X = np.zeros((len(training_patients),11))

    dates = []
    regi = []
    birth = []
    for i in range(len(training_patients)):
        get_patient = list(filter(lambda x: x['Patient Id'] == training_patients[i]['Patient Id'] and x['\ufeffHospital Id'] == training_patients[i]['Hospital Id'], data_patients))
        Y[i][0] = training_patients[i]['No Show/LateCancel Flag']
        appointment = get_patient[0]['Procedure Date'].split('/')
        dates.append(date(2000 + int(appointment[2]), int(appointment[0]), int(appointment[1])))
        made_app = get_patient[0]['Registration Date'].split('/')
        regi.append(date(2000 + int(made_app[2]), int(made_app[0]), int(made_app[1])))
        age = get_patient[0]['Date of Birth'].split('/')
        if len(age) == 1:
            birth.append(None)
        else:
            birth.append(date(1900 + int(age[2]), int(age[0]), int(age[1])))
        X[i,0] = 1. if get_patient[0]['Email'] == 'TRUE' else 0.
        X[i,1] = 1. if get_patient[0]['SMS'] == 'TRUE' else 0.
    X = getInfo(X, training_patients, dates, birth, regi, scale)
    return X, Y

def getValidationData(scale=None):
    X_validation = np.zeros((len(validation_patients),11))

    dates = []
    birth = []
    regi = []
    for i in range(len(validation_patients)):
        get_patient = list(filter(lambda x: x['Patient Id'] == validation_patients[i]['Patient Id'] and x['\ufeffHospital Id'] == validation_patients[i]['Hospital Id'], data_patients))
        appointment = get_patient[0]['Procedure Date'].split('/')
        day_of = date(2000 + int(appointment[2]), int(appointment[0]), int(appointment[1]))
        made_app = get_patient[0]['Registration Date'].split('/')
        regi.append(date(2000 + int(made_app[2]), int(made_app[0]), int(made_app[1])))
        dates.append(day_of)
        age = get_patient[0]['Date of Birth'].split('/')
        if len(age) == 1:
            birth.append(None)
        else:
            birth.append(date(1900 + int(age[2]), int(age[0]), int(age[1])))
        X_validation[i,0] = 1. if get_patient[0]['Email'] == 'TRUE' else 0.
        X_validation[i,1] = 1. if get_patient[0]['SMS'] == 'TRUE' else 0.
    return getInfo(X_validation, validation_patients, dates, birth, regi, scale)

def main():
    collectData()
    X_train, y_train = getTrainingData()
    X_val = getValidationData()
    np.save('xTrain', X_train)
    np.save('yTrain', y_train)
    np.save('xVal', X_val)
    X_train_norm, y_train = getTrainingData("norm")
    X_val_norm = getValidationData("norm")
    np.save('xTrain_norm', X_train_norm)
    np.save('xVal_norm', X_val_norm)
    X_train_minmax, y_train = getTrainingData("MinMax")
    X_val_minmax = getValidationData("MinMax")
    np.save('xTrain_minmax', X_train_minmax)
    np.save('xVal_minmax', X_val_minmax)

if __name__ == '__main__':
    main()
