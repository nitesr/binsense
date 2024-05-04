import { FunctionComponent } from 'react'
import { useNavigate } from 'react-router-dom'
import useLocalStorageState from 'use-local-storage-state'
import defaultImage from '../../assets/select_bin.png'
import classes from './binselector.module.scss'
import { CartProps, Product } from '../Products/Products'

// TODO: move all the logic to a logic component

export type Item = {
    product: Product
    status: string
}

interface BasketItem {
    prod_id: string
    quantity: number
}

interface Basket {
    items: BasketItem[]
}

interface FulfilRequest {
    basket: Basket
    bin_image: string
}

interface BasketItemFulfilStatus {
    basket_item: BasketItem
    status: string
}

interface FulfilResult {
    status: BasketItemFulfilStatus[]
}

export const BinSelector: FunctionComponent = () => {
    const [imageURL, setImageURL] = useLocalStorageState<string>('bin_image')
    const [imageBase64, setImageBase64] = useLocalStorageState<string>('bin_imagebase64');
    const [results, setResults] = useLocalStorageState<Item[]>('results');
    const [cart] = useLocalStorageState<CartProps>('cart', {})
    const navigate = useNavigate()

    const navigateToReview = () => {
        navigate('/review')
    }
    
    const handleImageSelect = (event: any) => {
        if (event.target.files?.item(0)) {
            // Get the file name
            const file = URL.createObjectURL(event.target.files[0]);
    
            // Update the imageURL state
            setImageURL(file);

            const reader = new FileReader();
            reader.onload = () => {
                const base64Image = reader.result as string;
                setImageBase64(base64Image);
            };
            reader.readAsDataURL(event.target.files[0]);
        }
    };

    const handleUpload = async () => {
        const definedCart = cart || {}

        if (imageBase64) {

            const items: BasketItem[] = Object.keys(definedCart).map((productId) => {
                return {
                    prod_id: productId,
                    quantity: definedCart[productId].quantity
                };
            });

            const basket: Basket = {
                items: items
            }

            const requestBody: FulfilRequest = {
                bin_image: imageBase64,
                basket: basket
            };
      
            try {
              const response = await fetch('/api/checkout', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
              });
      
              if (response.ok) {
                const responseData = await response.json();
                const result = responseData as FulfilResult;
                console.log('response from server:', result.status);

                const items: Item[] = result.status.map(itemStatus => {
                    return {
                      status: itemStatus.status,
                      product: definedCart[itemStatus.basket_item.prod_id]
                    };
                });
                setResults(items)

                console.log('results stored to localstorage: ', results);
                navigateToReview()

              } else {
                console.error('Failed to upload image');
              }
            } catch (error) {
              console.error('An error occurred:', error);
            }
          } else {
            console.error('No image selected');
          }
      };

    return (
        <section className={classes.binselector}>
          <h1>Bin Picker</h1>
    
          <div className={classes.container}>
            <div className={classes.bin}>
                <img src={imageURL ? imageURL : defaultImage} alt="Image" onClick={() => {
                    // Prompt the user to upload an image
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.onchange = handleImageSelect;
                    input.click();
                }}/>
            </div>
            <button onClick={handleUpload}>Validate</button>
          </div>
        </section>
      )
}