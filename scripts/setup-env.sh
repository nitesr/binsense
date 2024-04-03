#!/usr/bin/env bash

function set_dirs () {
    local SOURCE=${BASH_SOURCE[0]}
    while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
        local DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
        SOURCE=$(readlink "$SOURCE")
        [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
    done
    CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
}

function usage() {
    echo "$1 [--reset] [--clean] [--file my-condaenv.yml] [--help] "
    echo "options: "
    echo "  --reset : Optional. deletes and creates the $2 conda environment"
    echo "  --clean: Optional. deletes the $2 conda environment"
    echo "  --file my-condaenv.yml: Optional. uses binsense/my-condaenv.yml for conda environment"
    echo "  --help: Optional. for usage"
    echo "examples: "
    echo "  $1 --reset --file $(whoami)-condaenv.yml"
}

# provides CUR_DIR and SCRIPT_DIR variables
set_dirs
projDir="$(dirname "$SCRIPT_DIR")"
echo "script_dir=$SCRIPT_DIR"
echo "proj_dir=$projDir"
echo "cur_dir=$CUR_DIR"
if [ ! -d $projDir ]; then
    echo "couldn't find the project directory @ $projDir"
fi

condaFileName="conda-environment.yml"
echo "condaFileName=$condaFileName"

CREATE_ENV=1
CLEAN_ENV=0
HELP=0

OPTIND=1 
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help) HELP=1 ;;
    -c|--clean) CLEAN_ENV=1; CREATE_ENV=0 ;;
    -r|--reset) CLEAN_ENV=1 ;;
    -f|--file) condaFileName="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

condaFilePath=$projDir/$condaFileName
echo "condaFilePath=$condaFilePath"
if [ ! -f $condaFilePath ]; then
    echo "couldn't find the conda environment file @ $condaFilePath"
fi

envName=$(head -1 $condaFilePath | cut -d ':' -f 2 | tr -d '[:space:]')
pythonVer=$(grep "python=" $condaFilePath | cut -d '=' -f 2 | tr -d '[:space:]')

if [ $HELP -eq 1 ]; then
    usage "$0" "$envName"
    exit 0
fi

echo "project dir: $projDir"
echo "virtual environment: $envName"
echo "Python version: $pythonVer"

envAvailable=$( conda env list | grep $envName | wc -l )
if [ $envAvailable -eq 1 ] && [ $CLEAN_ENV -eq 1 ]; then
    echo "deleting $envName conda environment"
    conda deactivate
    conda env remove -n $envName
fi

envAvailable=$(conda env list | grep $envName | wc -l)
if [ $envAvailable -eq 0 ] && [ $CREATE_ENV -eq 1 ]; then
    echo "creating conda environment"
    conda env create -f $condaFilePath
    if [ $? -eq 0 ]; then
        envAvailable=1
    fi
fi

if [ $envAvailable -eq 1 ]; then
    conda activate $envName
    # conda activate $envDir
    if [ $? -eq 0 ]; then
        conda info --envs
        echo "pip @ $(which pip)"
        if [ ! -d "$projDir/dev-requirements.txt" ]; then
            pip install -r $projDir/dev-requirements.txt
        fi

        if [ ! -d "$projDir/requirements.txt" ]; then
            pip install -r $projDir/requirements.txt
        fi
    fi

    echo "Environment: $envName"
    echo "you can issue command 'conda activate $envName' to activate in the terminal"
fi
exit 0
