import { FunctionComponent, useEffect, useState } from 'react'
import useLocalStorageState from 'use-local-storage-state'
import ReactPaginate from 'react-paginate'

import classes from './products.module.scss'
import { Loader } from '../Loader'
import ProductCard from './ProductCard.tsx'
import type { Product } from './ProductCard.tsx'

const API_URL = '/api/products'

export type CartItem = {
  product: Product
  quantity: number
};

export interface CartProps {
  [productId: string]: CartItem
}

interface SearchParams {
  searchTerm: string;
  offset: number;
  limit: number;
}

interface Response {
  total: number;
  results: Product[];
}

export const Products: FunctionComponent = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [products, setProducts] = useState<Product[]>([]);
  const [offset, setOffset] = useState<number>(0);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [total, setTotal] = useState<number>(0);

  const [error, setError] = useState(false)
  const [cart, setCart] = useLocalStorageState<CartProps>('cart', {})

  const fetchProducts = async (params: SearchParams) => {
    try {
      const response = await fetch(`${API_URL}?prod_name=${params.searchTerm}&skip=${params.offset}&limit=${params.limit}`);
      if (response.ok) {
        const data: Response = await response.json();
        setProducts(data.results)
        setTotal(data.total)
        setIsLoading(false)
      }  else {
        setError(true)
        setIsLoading(false)
      }
    } catch (error) {
      setError(true)
      setIsLoading(false)
    }
  };

  const handleSearch = () => {
    setOffset(0)
    fetchProducts({searchTerm, offset: 0, limit: 10})
  }

  useEffect(() => {
    fetchProducts({ searchTerm, offset, limit: 10 });
  }, [offset])

  const addToCart = (product: Product):void => {
    const item: CartItem = {
      product: product,
      quantity: 1
    }
    setCart((prevCart) => ({
      ...prevCart,
      [product.id]: item,
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
      <input value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Search term" />
      <button onClick={handleSearch}>Search</button>
 
      <div className={classes.container}>
        {products &&
          products.map((product: Product) => (
            <ProductCard key={product.id} product={product} isInCartfn={isInCart} addToCartfn={addToCart} />
          ))}

        <ReactPaginate
          activeClassName={`${classes.item} ${classes.active}`}
          breakLabel={'...'}
          containerClassName={classes.pagination}
          disabledClassName={classes.disabled_page}
          nextClassName={`${classes.item} ${classes.next}`}
          pageClassName={`${classes.item} ${classes.pagination_page}`}
          previousClassName={`${classes.item} ${classes.previous}`}
          nextLabel="next >"
          onPageChange={(data) => setOffset(data.selected * 10)}
          pageRangeDisplayed={5}
          pageCount={total/10}
          previousLabel="< previous"
          marginPagesDisplayed={2}
          renderOnZeroPageCount={null}
        />
      </div>
    </section>
  )
}
