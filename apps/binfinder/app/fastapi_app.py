import PIL.Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

from binsense.model_api import ModelApi, OwlImageQuerier, OwlBinPreprocessor
from binsense.owlv2.config import Owlv2Config
from binsense.owlv2 import hugg_loader as hloader
from binsense.owlv2.model import Owlv2ForObjectDetection
from binsense.embed_datastore import EmbeddingDatastore, SafeTensorEmbeddingDatastore
    
from typing import List, Optional
from enum import Enum
from io import BytesIO

from pydantic import BaseModel
import pandas as pd
import base64, os, PIL, logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    PRODUCT_CSV_PATH: Optional[str] = None
    EMBEDDED_PRODUCT_CSV_PATH: Optional[str] = None
    USE_OWL: bool
    USE_OWL_BASELINE: bool
    OWL_MODEL_PATH: Optional[str] = None
    EMBED_STORE_DIR: Optional[str] = None
    BIN_IMAGES_DIR: Optional[str] = None
    TORCH_DEVICE: str = 'cpu'
    MAX_BASKET_LEN: int = 4

config = Settings()

def load_products() -> pd.DataFrame:
    csvpath = config.EMBEDDED_PRODUCT_CSV_PATH if config.USE_OWL else config.PRODUCT_CSV_PATH
    products_df = pd.read_csv(csvpath, dtype={'item_id': str})
    products_df.rename(columns={"item_id" : "id", "item_name": "name"}, inplace=True)
    print(f"loaded products of size {products_df.shape}")
    return products_df

def _get_owl_baseline_model() -> OwlImageQuerier:
    owl_model_cfg = Owlv2Config(**hloader.load_owlv2model_config())
    model = Owlv2ForObjectDetection(owl_model_cfg)
    model.load_state_dict(hloader.load_owlv2model_statedict())
    return OwlImageQuerier(model=model, threshold=0.998, device=config.TORCH_DEVICE)

def build_owl_model() -> ModelApi:
    owl_model = _get_owl_baseline_model()
    preprocessor = OwlBinPreprocessor(owl_model.processor)
    embed_ds = SafeTensorEmbeddingDatastore(config.EMBED_STORE_DIR, read_only=True).to_read_only_store()
    return ModelApi(
        model=owl_model, 
        preprocessor=preprocessor,
        embed_ds=embed_ds,
        bin_images_dir=config.BIN_IMAGES_DIR
    )

def build_model() -> ModelApi:
    if config.USE_OWL and config.USE_OWL_BASELINE:
        return build_owl_model()
    raise ValueError('all other model options not supported yet!')

products_df = load_products()
model = build_model()

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
    pred_qty: Optional[int] = None

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
    df = df.iloc[skip : end_idx]
    return SearchResults(
        offset=skip, total=total, 
        results= [ Product(**kwargs) for kwargs in df.to_dict(orient='records')]
    )

@app.post("/api/checkout")
def checkout(req: FulfilRequest):
    try :

        if len(req.basket.items) > config.MAX_BASKET_LEN:
            raise ValueError(f"basket length exceeded. {len(req.basket.items)} > {config.MAX_BASKET_LEN}")

        f = req.bin_image
        img = base64.b64decode(f.split('base64')[1])
        bin_image = PIL.Image.open(BytesIO(img))

        item_ids = [ item.prod_id for item in req.basket.items ]
        pred_qts, pred_boxes, ann_image = model.find_item_qts_in_binimage(item_ids=item_ids, bin_image=bin_image)

        buffer = BytesIO()
        ann_image.save(buffer, format="JPEG")
        ann_image_bytes = base64.b64encode(buffer.getvalue())
        ann_image_base64 = bytes("data:image/jpeg;base64,", encoding='utf-8') + ann_image_bytes

        fr = FulfilResult(bin_image=str(ann_image_base64, encoding='utf-8'), status=[])
        for i, item in enumerate(req.basket.items):
            status = FulfilStatus.NONE
            if item.quantity <= pred_qts[i]:
                status = FulfilStatus.FULL
            elif pred_qts[i] > 0:
                status = FulfilStatus.PARTIAL

            fr.status.append(
                BasketItemFulfilStatus(
                    basket_item=item, 
                    status=status,
                    pred_qty=pred_qts[i]
                ))
        return fr
    
    except PIL.UnidentifiedImageError:
        raise ValueError('badImage, expects of format data:image/jpeg;base64,<data>')
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    