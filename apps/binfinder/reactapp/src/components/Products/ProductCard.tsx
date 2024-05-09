import React from 'react';
import productIcon from '../../assets/product_icon.jpeg';
import classes from './products.module.scss';

export type Product = {
    id: number
    name: string
    image: string
};

const ProductCard: React.FC<{ product: Product, isInCartfn: Function, addToCartfn: Function }> = ({ product, isInCartfn, addToCartfn }) => {
  return (
    <div className={classes.product} key={product.id}>
        <img src={product.image ? product.image : productIcon} alt={product.name} />
        <h5>{product.name}</h5>
        <button disabled={isInCartfn(product.id)} onClick={() => addToCartfn(product)}>Add to Cart</button>
    </div>
  );
};

export default ProductCard;