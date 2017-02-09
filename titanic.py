# Import the Pandas library
import pdb
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# Load the train and test datasets to create two DataFrames
train_url = "train.csv"
train = pd.read_csv(train_url)

test_url = "test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

test.Fare = test.Fare.fillna(train.Fare.median())
train.Fare = train.Fare.fillna(train.Fare.median())

train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
 	
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')
test["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18 ] = 1 
train["Child"][train["Age"] >= 18 ] = 0
test["Child"][test["Age"] < 18 ] = 1 
test["Child"][test["Age"] >= 18 ] = 0

# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')
test["Embarked"] = test["Embarked"].fillna('S')
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2



def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 0,
                           np.where((i['SibSp']+i['Parch']) <= 3,1, 2))
        del i['SibSp']
        del i['Parch']
    return train, test

train, test = fam_size(train,test)


#pdb.set_trace()
train.isnull().any()
#Print the Sex and Embarked columns


# =========================			TARGET 			===================================#

target = train['Survived'].values

# =========================			TRAIN 			===================================#


# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "Embarked", "Name_Len", "Fam_Size"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
print(target)
my_forest = forest.fit(features_forest, target)

#pdb.set_trace()

print(my_forest.feature_importances_)
# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# =========================			TEST 			===================================#



# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "Embarked", "Name_Len", "Fam_Size"]].values
pred_forest = my_forest.predict(test_features)

print(len(pred_forest))


PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])
