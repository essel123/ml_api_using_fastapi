import pickle
import joblib
import pandas as pd
from fastapi import FastAPI

from pydantic import BaseModel


class Data(BaseModel):
    Age: int
    Number_of_sexual_partners: float
    First_sexual_intercourse: float
    Num_of_pregnancies: float
    Smokes: float
    Smokes_Years: float
    Smokes_pack_per_year: float
    Hormonal_Contraceptives: float
    Hormonal_Contraceptives_years: float
    IUD: float
    IUD_years: float
    STDs: float
    Number_of_diagnosis: float
    Time_since_last_diagnosis: float


model = joblib.load('model.pkl')

# print(model.predict([[71,3,17,6,1,34,3.4,0,0,1,7,0,0,3]])[0])

app = FastAPI()


@app.post('/')
async def data_endpoint(item: Data):

    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())

    biospy= model.predict(df)

    return {"Biospy Results": int(biospy)}


