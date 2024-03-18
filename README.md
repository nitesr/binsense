# binsense
This project includes necessary work to address the capstone Project 4 done part of IK ML Switch up program. The project is inspired from the [Amazon bin challenge](https://github.com/silverbottlep/abid_challenge)

## Project-4: BinSense AI - Inventory Vision System
Objectives: \
Use the dataset (images + metadata) and develop a highly accurate and fast computer vision model to verify if the items with their respective quantities are present in the image of the bin. 

For example, You get an order for 3 items and their quantities as mentioned below.You get an image of the bin where items in the order are present. Your objective is to validate whether the items in the order are the ones in the bin.
![flow](./docs/objective_flow.png)


Dataset : [drive](https://docs.google.com/spreadsheets/d/1rZfFrHEbfX_b-3ofEIDxLQUFNDWtmDarXBrU4nn4Luw/edit#gid=601918728), [amazon dataset](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)

## Project contributors
| Name[^1]            | email                  |
|---------------------|------------------------|
| Leo Pimentel        | leo@groundswellai.com  |
| Nitesh Chinthireddy | reddy.nitesh@gmail.com |
| Rathi anand         | rathianandk@gmail.com  |
[^1]: *alphabetically ordered*

## Tools
- [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Podman or Docker
- VScode (code editor)
- Juypter

## Languages
- Python
- Shell
- Javascript
- HTML
- CSS

## Project structure
|-binsense (root dir) \
|-|--scripts    : scripts to setup env, etc. \
|-|--docs       : detailed documents about objective, data analysis, data prep, model arch etc. \
|-|--libs       : shared code used in apps & notebooks \
|-|--notebooks  : try out juyptner notebooks \
|-|--apps       : deployable apps, docker files and necessary scripts

## Project setup
Sets up conda environment with python 3.11 & dependencies listed in train-environment.yml (using conda), *requirements.txt (using pip).

For API, we will define docker file to setup the environment.
```
./scripts/setup-env.sh --help
./scripts/setup-env.sh
conda activate binsense_condaenv
```


