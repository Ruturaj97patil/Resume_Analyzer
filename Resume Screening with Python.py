#!/usr/bin/env python
# coding: utf-8

# In[137]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[138]:


df = pd.read_csv('UpdatedResumeDataSet.csv')


# In[139]:


df.head()


# In[140]:


df.shape


# In[141]:


sns.countplot(df['Category'])


# # Exploring Categories

# In[142]:


df['Category'].value_counts()


# In[143]:


df['Category'] = df['Category'].astype('category')


# In[144]:


sns.countplot(data=df, x='Category')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[145]:


plt.figure(figsize=(15, 5))
sns.countplot(data=df, x='Category')
plt.xticks(rotation=90)
plt.show()


# In[146]:


df['Category'].unique()


# In[147]:


counts = df['Category'].value_counts()
labels = df['Category'].unique()

colors = plt.cm.plasma(np.linspace(0, 1, 3))

plt.figure(figsize=(15,10))
plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True,colors=colors) 
plt.title('Distribution of Categories')
plt.show()


# # Exploring Resume

# In[148]:


df['Category'][0]


# In[149]:


df['Resume'][0]


# # Cleaning Data:
# 1. URLs,
# 2. hashtags,
# 3. mentions,
# 4. special letters,
# 5. punctuations:

# In[150]:


import re
def cleanResume(txt):
    cleanText = re.sub(r'http\S+', ' ', txt)
    cleanText = re.sub(r'\bRT\b|\bcc\b', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    cleanText = re.sub(r'[^\w\s]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7F]+', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText


# In[151]:


cleanResume("my ###### $ #fasf website like is this http://heloworld and access it @gmain.com")


# In[152]:


df['Resume'] = df['Resume'].apply(lambda x:cleanResume(x))


# In[153]:


df['Resume'][0]


# # Words into categorical values

# In[154]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[155]:


le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])


# In[156]:


df.Category.unique()


# In[157]:


# ['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing', ..., 'Hadoop', 'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
# Length: 25
# Categories (25, object): ['Advocate', 'Arts', 'Automation Testing', 'Blockchain', ..., 'SAP Developer', 'Sales', 'Testing', 'Web Designing']


# #  Vactorization

# In[158]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(df['Resume'])
requiredText= tfidf.transform(df['Resume'])


# In[159]:


requiredText


# # Splitting

# In[160]:


from sklearn.model_selection import train_test_split


# In[161]:


X_train,X_test,y_train,y_test = train_test_split(requiredText,df['Category'], test_size = 0.2, random_state=42)


# In[162]:


X_train.shape


# In[163]:


X_test.shape


# # Now let's train the model and print the classification report:

# In[164]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train,y_train)
ypred = clf.predict(X_test)
print(accuracy_score(y_test,ypred))


# In[165]:


myresume = """Contact
Education
+91 8976505110
Phone
ruturajsadashivpatil727@gmail.com
Email
Tirupati Garden plot no 103 flat no 301 sec 34, Kamothe, Navi Mumbai 410209
Address
Ruturaj Patil
Enthusiastic about applying skills and knowledge in a real-world setting, and eager to contribute to a dynamic team while gaining valuable hands-on experience.
Projects
The "Medical Insurance Calculation System Using Python" is a comprehensive software application that leverages the Python programming language to assist healthcare providers and patients in calculating medical insurance costs. This system integrates data input, processing, and analysis capabilities to determine insurance premiums,
The QR Code Generator project is a web-based application developed using HTML, CSS, and JavaScript. This project enables users to create QR codes for various purposes quickly and easily. Users input the desired content, such as a URL, text, or contact information, and the application generates a QR code image that can be downloaded.
AptiPro Placement Web Application
Medical Insurance Calculation System Using Python
QR CODE GENERATOR
July 2023- Nov 2023
May 2023- July 2023
July 2022- Aug 2022
laxmi.gadhikar@it.fcrit.ac.in
Dr. Laxmi Gadhikar
Email :
Assistant Professor, FCRIT VASHI
shubhangi.vaikole@it.fcrit.ac.in
Dr. Shubhangi Vaikole
Email :
Hod of IT Dept, FCRIT VASHI
Reference
FCRIT, Vashi University of Mumbai
Mahatma Jr.College of science, Raigad
BE IT  CGPI- 8.73
12th       Grade- 81%
2021-2025
2019-2020
HTML
Communication, Leadership
Javascript
Python
PHP
CSS
English
Expertise
Language
Hindi
Marathi
The project aims to develop a user-friendly Multiple-Choice Question (MCQ) test website tailored for conducting placement aptitude tests in colleges. It offers educators the ability to curate tests by selecting questions from a structured question bank, ensuring students possess the required skills for successful placements.
Aspiring Data Science/Web Development Intern
"""


# In[166]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'cleanResume' and 'df' are already defined

# Create the TfidfVectorizer
tfidfd = TfidfVectorizer(stop_words='english')

# Fit and transform the 'Resume' column of the DataFrame
tfidf_matrix = tfidfd.fit_transform(df['Resume'])


import pickle
pickle.dump(tfidfd,open('tfidf.pkl','wb'))
pickle.dump(clf,open('clf.pkl','wb'))


# In[167]:


myresume = """Contact
Education
+91 8976505110
Phone
ruturajsadashivpatil727@gmail.com
Email
Tirupati Garden plot no 103 flat no 301 sec 34, Kamothe, Navi Mumbai 410209
Address
Ruturaj Patil
Enthusiastic about applying skills and knowledge in a real-world setting, and eager to contribute to a dynamic team while gaining valuable hands-on experience.
Projects
The "Medical Insurance Calculation System Using Python" is a comprehensive software application that leverages the Python programming language to assist healthcare providers and patients in calculating medical insurance costs. This system integrates data input, processing, and analysis capabilities to determine insurance premiums,
The QR Code Generator project is a web-based application developed using HTML, CSS, and JavaScript. This project enables users to create QR codes for various purposes quickly and easily. Users input the desired content, such as a URL, text, or contact information, and the application generates a QR code image that can be downloaded.
AptiPro Placement Web Application
Medical Insurance Calculation System Using Python
QR CODE GENERATOR
July 2023- Nov 2023
May 2023- July 2023
July 2022- Aug 2022
laxmi.gadhikar@it.fcrit.ac.in
Dr. Laxmi Gadhikar
Email :
Assistant Professor, FCRIT VASHI
shubhangi.vaikole@it.fcrit.ac.in
Dr. Shubhangi Vaikole
Email :
Hod of IT Dept, FCRIT VASHI
Reference
FCRIT, Vashi University of Mumbai
Mahatma Jr.College of science, Raigad
BE IT  CGPI- 8.73
12th       Grade- 81%
2021-2025
2019-2020
HTML
Communication, Leadership
Javascript
Python
PHP
CSS
English
Expertise
Language
Hindi
Marathi
The project aims to develop a user-friendly Multiple-Choice Question (MCQ) test website tailored for conducting placement aptitude tests in colleges. It offers educators the ability to curate tests by selecting questions from a structured question bank, ensuring students possess the required skills for successful placements.
Aspiring Data Science/Web Development Intern
"""


# In[172]:


import pickle

# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))

# Clean the input resume
cleaned_resume = cleanResume(myresume)

# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])

# Make the prediction using the loaded classifier
prediction_id = clf.predict(input_features)[0]

# Map category ID to category name
category_mapping = {
    15:"Java Developer",
    23:"Testing",
    8: "DevOps Engineer",
    20:"Python Developer",
    24:"Wev Designing",
    12:"HR",
    13:"Hadoop",
    3:"Blockchain",
    10:"ETL Developer",
    18:"Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11:"Electrical Engineering",
    14: "Health and Fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21 : "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

category_name = category_mapping.get(prediction_id, "Unknown")

print("Predicted Category:", category_name)
print(prediction_id)


# In[ ]:





# In[ ]:




