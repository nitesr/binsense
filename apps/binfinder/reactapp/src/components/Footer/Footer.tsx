import { FunctionComponent } from 'react'
import classes from "./footer.module.scss"

export const Footer: FunctionComponent = () => {
  return (
    <footer className={classes.footer} data-cy="footer">
      <ul/>
    </footer>
  )
}
