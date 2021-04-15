import pickle
import pandas as pd
from flask import Flask,flash,request,render_template
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import os.path
import openpyxl

def getBin(a,maxValue):
    return [0 if i!=a else 1 for i in range(1,maxValue+1)]
    
def dropColumns(data,columns = []):
    return data.drop(columns,axis = 1)

app=Flask(__name__)

#rendering html pages

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/info")
def info():
    return render_template("info.html")
    
@app.route("/predict_performance")
def predict():
    return render_template("predict.html")    

@app.route("/show_result",methods=['GET','POST'])
def get_result():
    sample = []
    if request.method=='POST':
        try:
            name = request.form.get('name')
            reg = request.form.get('reg_no')
            gender = int(request.form.get('gender'))
            lunch = int(request.form.get('lunch'))
            parent = int(request.form.get('parent_education'))
            group = int(request.form.get('ethnicity'))
            test_preparation_course = int(request.form.get('test_preparation'))
            math_score = int(request.form.get('math_score'))
            reading_score = int(request.form.get('reading_score'))
            writing_score = int(request.form.get('writing_score'))
            total_score = math_score + reading_score + writing_score
            
            if math_score<0 or math_score>100 or reading_score<0 or reading_score>100 or writing_score<0 or writing_score>100:
                raise Exception()
            gender = getBin(gender,2)
            parent = getBin(parent,6)
            group = getBin(group,5)
            test_preparation_course = getBin(test_preparation_course,2)
            lunch = getBin(lunch,2)
            
            sample.append(math_score)
            sample.append(reading_score)
            sample.append(writing_score)
            sample.append(total_score)
            sample.extend(gender)
            sample.extend(parent)
            sample.extend(test_preparation_course)
            sample.extend(lunch)
            sample = [sample]
            
            #loading the data
            model=pickle.loads(open('model.pkl',"rb").read())
    
            if model.predict(sample)[0][0]==1:
                return render_template("fail.html")
            else:
                return render_template("pass.html")
        except:
            return render_template("error.html")
    return None

@app.route("/show_file_result",methods=['GET','POST'])
def get_csv_result():
    sample = []
    if request.method=='POST':
        try:
            file=request.files['csv_file']
            basepath=os.path.dirname(__file__)   #store the file dir
            filepath=os.path.join(basepath,"uploads",file.filename) #stores the the file in uploads folder
            file.save(filepath) #saving the file
            
            #reading uploaded file
            if file.filename[len(file.filename)-4:] == ".csv":
                data=pd.read_csv(filepath)
            else:
                data=pd.read_excel(filepath,engine="openpyxl")
                #data.to_csv("uploads/"+file.filename[:len(file.filename)-5]+".csv",index=None,header=True)
                #data=pd.read_csv("uploads/"+file.filename[:len(file.filename)-5]+".csv")
            
            #data preprocessing
            X = dropColumns(data,["Student_Id","race/ethnicity"])
            
            X["math score"] = X["math score"].astype(int)
            X["reading score"] = X["reading score"].astype(int)
            X["writing score"] = X["writing score"].astype(int)
            X["Total Score"] = X["Total Score"].astype(int)
            
            if not X["math score"].between(0,100).all() or not X["reading score"].between(0,100).all() or not X["writing score"].between(0,100).all() or not X["Total Score"].between(0,300).all():
                raise Exception()
                
            gender = list(X["gender"])
            X["gender_female"] = [1 if i=="female" else 0 for i in gender]
            X["gender_male"] = [1 if i=="male" else 0 for i in gender]
            
            parent = list(X["parental level of education"])
            X["parental level of education_associate's degree"] = [1 if i=="associate's degree" else 0 for i in parent]
            X["parental level of education_bachelor's degree"] = [1 if i=="bachelor's degree" else 0 for i in parent]
            X["parental level of education_high school"] = [1 if i=="high school" else 0 for i in parent]
            X["parental level of education_master's degree"] = [1 if i=="master's degree" else 0 for i in parent]
            X["parental level of education_some college"] = [1 if i=="some college" else 0 for i in parent]
            X["parental level of education_some high school"] = [1 if i=="some high school" else 0 for i in parent]
            
            test_preparation_course = list(X["test preparation course"])
            X["test preparation course_completed"] = [1 if i=="completed" else 0 for i in test_preparation_course]
            X["test preparation course_none"] = [1 if i=="none" else 0 for i in test_preparation_course]
            
            lunch = list(X["lunch"])
            X["lunch_free/reduced"] = [1 if i=="free/reduced" else 0 for i in lunch]
            X["lunch_standard"] = [1 if i=="standard" else 0 for i in lunch]
            
            X = dropColumns(X,["gender","parental level of education","lunch","test preparation course"])
            
            #loading the model data
            model=pickle.loads(open('model.pkl',"rb").read())
            
            X = (X.values).tolist()
            
            output = model.predict(X)
            output = ['no' if i[0]==1 else 'yes' for i in output]
            
            data['Pass/Fail'] = output
            data = dropColumns(data,["gender","parental level of education","lunch","test preparation course","math score","reading score","writing score","Total Score","race/ethnicity"])
                    
            return render_template("file_output.html",dataframe=data,columns=list(data.columns),values=list(data.values))
        except:
            return render_template("error.html")
    return None
    
    
# MAin Function

if __name__ == "__main__":
    app.run()
    
