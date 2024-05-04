from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware

from typing import List
from enum import Enum

from pydantic import BaseModel
import base64

class Product(BaseModel):
    id: str
    name: str
    image: str = None

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
    status: List[BasketItemFulfilStatus]

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")
# app.add_middleware(
#   CORSMiddleware,
#   allow_origins = ["*"],
#   allow_methods = ["*"],
#   allow_headers = ["*"]
# )

@app.get('/', include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/api/products/{prod_id}')
def get_products(prod_id: str = None) -> Product:
    raise HTTPException(status_code=404, detail='prod_not_found')

@app.get('/api/products')
def get_products(prod_name: str = None) -> List[Product]:
    return [
        Product(id="1", name="Product 1", image=None), 
        Product(id="2", name="Product 2", image=None)
    ]

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