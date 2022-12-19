from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

import pandas as pd
import numpy as np
import random

import pickle

random.seed(42)
np.random.seed(42)
#--------------------------------------------------------------------------
app = FastAPI()

class Item(BaseModel):
    name: str
    year: Union[int, None] = None
    selling_price: int
    km_driven: Union[int, None] = None
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[str, None] = None
    engine: Union[str, None] = None
    max_power: Union[str, None] = None
    torque: Union[str, None] = None
    seats: Union[float, None] = None

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item): #-> float:

    df = pd.DataFrame([item.dict()])
# --------------------------------------------------------------------------
# Кастуем поля, убираем е.и.
    if df.mileage.values[0] != None:
        df.mileage = df.mileage.str.replace('^$', 'nan', regex=True)
        df.mileage = df.mileage.map(lambda x: float(str(x).rstrip(' kmpl').rstrip(' km/kg')))
        df.mileage = df.mileage.astype('float64')
    else:
        df.mileage = df.mileage.astype('str').str.replace('None', 'nan', regex=True).astype('float64')

    if df.engine.values[0] != None:
        df.engine = df.engine.str.replace('^$', 'nan', regex=True)
        df.engine = df.engine.map(lambda x: float(str(x).rstrip(' CC')))
        df.engine = df.engine.astype('float64')
    else:
        df.engine = df.engine.astype('str').str.replace('None', 'nan', regex=True).astype('float64')

    if df.max_power.values[0] != None:
        df.max_power = df.max_power.str.replace('^$', 'nan', regex=True)
        df.max_power = df.max_power.map(lambda x: float(str(x).rstrip(' bhp').rstrip('bhp')))
        df.max_power = df.max_power.astype('float64')
    else:
        df.max_power = df.max_power.astype('str').str.replace('None', 'nan', regex=True).astype('float64')

    df.torque = None

# --------------------------------------------------------------------------
# Заполняем пропуски медианами
    medians = pd.read_csv('medians.csv')
    for col in medians.columns:
        if df[col].isna().any():
            df[col].loc[df[col].isna()] = medians[col]

# --------------------------------------------------------------------------
# Кастуем к инту
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)

# --------------------------------------------------------------------------
# Кодируем признаки
    numerical_cols = df.columns[df.dtypes != object]
    X_real = df[numerical_cols].drop(columns='selling_price')

    X_cat = df.drop(columns=['year','km_driven', 'mileage', 'engine','max_power','seats','torque', 'selling_price', 'name'])
    enc = pickle.load(open('ONE.pckl', mode='rb'))
    enc.transform(X_cat).toarray()
    X_cat_coded = pd.DataFrame(enc.transform(X_cat).toarray(), columns=enc.get_feature_names_out())
    X_cat_coded = X_cat_coded[X_cat_coded.columns[:-1]]

    df_coded = pd.concat([X_real,X_cat_coded], axis=1)

# --------------------------------------------------------------------------
# Модель
    ridge_reg = pickle.load(open('ridge_hw_1_alpha_506.pckl', mode='rb'))
    y_pred = ridge_reg.predict(df_coded.to_numpy())

    return pd.DataFrame(y_pred).to_dict()[0][0]

#-----------------------------------------------------------------------------------------------------------------------

@app.post("/predict_items")
def predict_items(items: List[Item]): #-> List[float]:

    df = pd.DataFrame([items[0].dict()])
    preds = [predict_item(items[0])]
    for item in items[1:]:
        preds.append(predict_item(item))
        to_add = pd.DataFrame([item.dict()])
        df = pd.concat([df, to_add])

    df.reset_index(inplace=True, drop=True)

    df = pd.concat([df, pd.Series(preds, name='y_pred')], axis=1)
    return df.to_dict()
