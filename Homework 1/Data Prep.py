# Imports
import pandas as pd
from sklearn.model_selection import train_test_split

#-----------
# Load data
#-----------

# Get data
full_raw = pd.read_csv("data/some data.csv")
# print("full : " + str(full_raw.shape)) (1460, 81)

# Check for duplicates: no duplicates

#-------------
# Pre-process
#-------------

# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"
full_raw.loc[:, "Alley"] = full_raw.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
full_raw.loc[:, "BedroomAbvGr"] = full_raw.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
full_raw.loc[:, "BsmtQual"] = full_raw.loc[:, "BsmtQual"].fillna("No")
full_raw.loc[:, "BsmtCond"] = full_raw.loc[:, "BsmtCond"].fillna("No")
full_raw.loc[:, "BsmtExposure"] = full_raw.loc[:, "BsmtExposure"].fillna("No")
full_raw.loc[:, "BsmtFinType1"] = full_raw.loc[:, "BsmtFinType1"].fillna("No")
full_raw.loc[:, "BsmtFinType2"] = full_raw.loc[:, "BsmtFinType2"].fillna("No")
full_raw.loc[:, "BsmtFullBath"] = full_raw.loc[:, "BsmtFullBath"].fillna(0)
full_raw.loc[:, "BsmtHalfBath"] = full_raw.loc[:, "BsmtHalfBath"].fillna(0)
full_raw.loc[:, "BsmtUnfSF"] = full_raw.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
full_raw.loc[:, "CentralAir"] = full_raw.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
full_raw.loc[:, "Condition1"] = full_raw.loc[:, "Condition1"].fillna("Norm")
full_raw.loc[:, "Condition2"] = full_raw.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
full_raw.loc[:, "EnclosedPorch"] = full_raw.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
full_raw.loc[:, "ExterCond"] = full_raw.loc[:, "ExterCond"].fillna("TA")
full_raw.loc[:, "ExterQual"] = full_raw.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
full_raw.loc[:, "Fence"] = full_raw.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
full_raw.loc[:, "FireplaceQu"] = full_raw.loc[:, "FireplaceQu"].fillna("No")
full_raw.loc[:, "Fireplaces"] = full_raw.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
full_raw.loc[:, "Functional"] = full_raw.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
full_raw.loc[:, "GarageType"] = full_raw.loc[:, "GarageType"].fillna("No")
full_raw.loc[:, "GarageFinish"] = full_raw.loc[:, "GarageFinish"].fillna("No")
full_raw.loc[:, "GarageQual"] = full_raw.loc[:, "GarageQual"].fillna("No")
full_raw.loc[:, "GarageCond"] = full_raw.loc[:, "GarageCond"].fillna("No")
full_raw.loc[:, "GarageArea"] = full_raw.loc[:, "GarageArea"].fillna(0)
full_raw.loc[:, "GarageCars"] = full_raw.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
full_raw.loc[:, "HalfBath"] = full_raw.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
full_raw.loc[:, "HeatingQC"] = full_raw.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
full_raw.loc[:, "KitchenAbvGr"] = full_raw.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
full_raw.loc[:, "KitchenQual"] = full_raw.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
full_raw.loc[:, "LotFrontage"] = full_raw.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
full_raw.loc[:, "LotShape"] = full_raw.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
full_raw.loc[:, "MasVnrType"] = full_raw.loc[:, "MasVnrType"].fillna("None")
full_raw.loc[:, "MasVnrArea"] = full_raw.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
full_raw.loc[:, "MiscFeature"] = full_raw.loc[:, "MiscFeature"].fillna("No")
full_raw.loc[:, "MiscVal"] = full_raw.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
full_raw.loc[:, "OpenPorchSF"] = full_raw.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
full_raw.loc[:, "PavedDrive"] = full_raw.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
full_raw.loc[:, "PoolQC"] = full_raw.loc[:, "PoolQC"].fillna("No")
full_raw.loc[:, "PoolArea"] = full_raw.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
full_raw.loc[:, "SaleCondition"] = full_raw.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
full_raw.loc[:, "ScreenPorch"] = full_raw.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
full_raw.loc[:, "TotRmsAbvGrd"] = full_raw.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
full_raw.loc[:, "Utilities"] = full_raw.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
full_raw.loc[:, "WoodDeckSF"] = full_raw.loc[:, "WoodDeckSF"].fillna(0)

# Some numerical features are actually really categories
full_raw = full_raw.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45",
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75",
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120",
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })

# Encode some categorical features as ordered numbers when there is information in the order
full_raw = full_raw.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )


# Differentiate numerical features (minus the target) and categorical features
categorical_features = full_raw.select_dtypes(include = ["object"]).columns
numerical_features = full_raw.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop(["SalePrice","Id"])
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
full_raw_num = full_raw[numerical_features]
full_raw_cat = full_raw[categorical_features]

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in full_raw : " + str(full_raw_num.isnull().values.sum()))
full_raw_num = full_raw_num.fillna(full_raw_num.median())
print("Remaining NAs for numerical features in full_raw : " + str(full_raw_num.isnull().values.sum()))

# Create dummy features for categorical values via one-hot encoding
print("NAs for categorical features in full_raw : " + str(full_raw_cat.isnull().values.sum()))
full_raw_cat = pd.get_dummies(full_raw_cat)
print("Remaining NAs for categorical features in full_raw : " + str(full_raw_cat.isnull().values.sum()))

# Join categorical and numerical features
full_X = pd.concat([full_raw_num, full_raw_cat], axis = 1)
print("New number of features : " + str(full_X.shape[1]))
