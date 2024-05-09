from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
    
from typing import List
from enum import Enum

from pydantic import BaseModel
import pandas as pd
import base64, os

PRODUCT_CSV_PATH = f'{os.getenv("DATA_DIR")}/bin/products.csv' if 'DATA_DIR' in os.environ else '/data/bin/products.csv'
products_df = pd.read_csv(PRODUCT_CSV_PATH, dtype={'item_id': str})
products_df.rename(columns={"item_id" : "id", "item_name": "name"}, inplace=True)

class Product(BaseModel):
    id: str
    name: str
    image: str = None

class SearchResults(BaseModel):
    offset: int
    total: int
    results: List[Product]

class BasketItem(BaseModel):
    prod_id: str
    quantity: int

class Basket(BaseModel):
    items: List[BasketItem]
    
class Bin(BaseModel):
    image: str

class FulfilRequest(BaseModel):
    basket: Basket
    bin_image: str

class DataTag(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'

class FulfilStatus(str, Enum):
    FULL = 'full'
    PARTIAL = 'partial'
    NONE = 'none'

class BasketItemFulfilStatus(BaseModel):
    basket_item: BasketItem
    status: FulfilStatus

class FulfilResult(BaseModel):
    bin_image: str
    status: List[BasketItemFulfilStatus]

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")
app.add_middleware(
  CORSMiddleware,
  allow_origins = ["*"],
  allow_methods = ["*"],
  allow_headers = ["*"]
)

@app.get('/', include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/api/products/{prod_id}')
def get_products(prod_id: str = None) -> Product:
    if prod_id:
        df = products_df.query(f'id == "{prod_id}"')
        if df.shape[0] > 0:
            return Product(**(df.loc[0].to_dict()))

    raise HTTPException(status_code=404, detail='prod_not_found')

@app.get('/api/products')
def get_products(prod_name: str = None, skip: int = 0, limit: int = 10) -> SearchResults:
    
    df = products_df
    if prod_name and len(prod_name) > 0:
        mask = products_df['name'].str.contains(prod_name, regex=True, case=False)
        df = products_df[mask]
    
    if df.shape[0] == 0 or skip > df.shape[0]-1:
        return SearchResults(offset=skip, total=0, results=[])
    
    total = df.shape[0]
    end_idx = min(skip + limit, df.shape[0])
    df = df.iloc[skip : end_idx-1]
    return SearchResults(
        offset=skip, total=total, 
        results= [ Product(**kwargs) for kwargs in df.to_dict(orient='records')]
    )

@app.post("/api/checkout")
def checkout(req: FulfilRequest):
    try :
        f = req.bin_image
        img = base64.b64decode(f)

        fr = FulfilResult(status=[])
        for item in req.basket.items:
            fr.status.append(BasketItemFulfilStatus(basket_item=item, status=FulfilStatus.FULL))
        return fr
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))