import { FunctionComponent } from 'react'
import useLocalStorageState from 'use-local-storage-state'

import { Item } from '../BinSelector/BinSelector.tsx'
import productIcon from '../../assets/product_icon.jpeg'
import classes from './review.module.scss'

export const Review: FunctionComponent = () => {
  const [results] = useLocalStorageState<Item[]>('results')
  // const [imageURL, setImageURL] = useLocalStorageState<string>('bin_image')

  const getResults = () => Object.values(results || [])

  return (
    <section className={classes.review}>
      <h1>Results</h1>

      <div className={classes.container}>
        {getResults().map(result => (
          <div className={classes.product} key={result.product.id}>
            <img src={result.product.image ? result.product.image : productIcon} alt={result.product.name} />
            <h3>{result.product.name}</h3>
            <h3>{result.product.quantity}</h3>
            <h3>{result.status}</h3>
          </div>
        ))}
      </div>
    </section>
  )
}
