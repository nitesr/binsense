import { FunctionComponent, useEffect, useState } from 'react'
import useLocalStorageState from 'use-local-storage-state'

import productIcon from '../../assets/product_icon.jpeg'
import classes from './products.module.scss'
import { Loader } from '../Loader'

const API_URL = '/api/products'

export type Product = {
  id: number
  name: string
  image: string
  quantity: number
}

export interface CartProps {
  [productId: string]: Product
}

export const Products: FunctionComponent = () => {
  const [isLoading, setIsLoading] = useState(true)
  const [products, setProducts] = useState<Product[]>([])
  const [error, setError] = useState(false)
  const [cart, setCart] = useLocalStorageState<CartProps>('cart', {})


  useEffect(() => {
    fetchData(API_URL)
  }, [])


  async function fetchData(url: string) {
    try {
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setProducts(data)
        setIsLoading(false)
      } else {
        setError(true)
        setIsLoading(false)
      }
    } catch (error) {
      setError(true)
      setIsLoading(false)
    }
  }

  const addToCart = (product: Product):void => {
    product.quantity = 1

    setCart((prevCart) => ({
      ...prevCart,
      [product.id]: product,
    }))
  }

  const isInCart = (productId: number):boolean => Object.keys(cart || {}).includes(productId.toString())

  if (error) {
    return <h3 className={classes.error}>An error occurred when fetching data. Please check the API and try again.</h3>
  }

  if (isLoading) {
    return <Loader />
  }


  return (
    <section className={classes.productPage}>
      <h1>Products</h1>

      <div className={classes.container}>
        {products.map(product => (
          <div className={classes.product} key={product.id}>
            
            <img src={product.image ? product.image : productIcon} alt={product.name} />
            <h3>{product.name}</h3>
            <button disabled={isInCart(product.id)} onClick={() => addToCart(product)}>Add to Cart</button>
          </div>
        ))}
      </div>
    </section>
  )
}
