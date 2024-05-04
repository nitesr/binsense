import { FunctionComponent, useEffect } from 'react'
import useLocalStorageState from 'use-local-storage-state'

import { Quantifier } from '../Quantifier'
import { CartProps } from '../Products/Products.tsx'
import { Operation } from '../Quantifier/Quantifier.tsx'
import productIcon from '../../assets/product_icon.jpeg'
import classes from './cart.module.scss'
import { useLocation, useNavigate } from 'react-router-dom'


export const Cart: FunctionComponent = () => {
  const [cart, setCart] = useLocalStorageState<CartProps>('cart', {})
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [location])

  const handleRemoveProduct = (productId: number): void => {
    setCart((prevCart) => {
      const updatedCart = { ...prevCart }
      delete updatedCart[productId]
      return updatedCart
    })
  }

  const handleUpdateQuantity = (productId: number, operation: Operation) => {
    setCart((prevCart) => {
      const updatedCart = { ...prevCart }
      if (updatedCart[productId]) {
        if (operation === 'increase') {
          updatedCart[productId] = { ...updatedCart[productId], quantity: updatedCart[productId].quantity + 1 }
        } else {
          updatedCart[productId] = { ...updatedCart[productId], quantity: updatedCart[productId].quantity - 1 }
        }
      }
      return updatedCart
    })
  }
  

  const navigateToBinSelector = () => {
    navigate('/checkout')
  }


  const getProducts = () => Object.values(cart || {})

  return (
    <section className={classes.cart}>
      <h1>Cart</h1>

      <div className={classes.container}>
        {getProducts().map(product => (
          <div className={classes.product} key={product.id}>
            <img src={product.image ? product.image : productIcon} alt={product.name} />
            <h3>{product.name}</h3>
            <Quantifier
              removeProductCallback={() => handleRemoveProduct(product.id)}
              productId={product.id}
              handleUpdateQuantity={handleUpdateQuantity} />
          </div>
        ))}
        <button onClick={() => navigateToBinSelector()}>Checkout</button>
        
      </div>
    </section>
  )
}
