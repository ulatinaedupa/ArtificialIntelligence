from fastapi import FastAPI      # api development
from pydantic import BaseModel   # improve data api preparation
import dill                      # load model
import pandas as pd              # for data manipulation


api = FastAPI()                  # instantiate the API
with open('GB.pkl', 'rb') as f:  # load the model
    model = dill.load(f)

class ScoringItem(BaseModel):    # type check tru pydantic
    TransactionDate: str
    HouseAge: float
    DistanceToStation: float
    NumberOfPubs: float
    PostCode: str

@api.post('/')                   # route the endpoint 
async def scoring_endpoint(item:ScoringItem): # from the validated input in async mode 
    df = pd.DataFrame(data=[item.dict().values()], # create the dataframe for preds
                      columns=item.dict().keys())
    y_pred = model.predict(df)  # predict
    return {'prediction': round(float(y_pred), 2)} # return the prediction