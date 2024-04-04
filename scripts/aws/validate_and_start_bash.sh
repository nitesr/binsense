#!/bin/bash


function usage () {
    echo 'docker run -it {IMAGE_ID}" "AWS_PROFILE=<your-name-no-spaces>" "AWS_ACCESS_KEY=$IK_USER_AWS_ACCESS_KEY" "AWS_ACCESS_SECRET=$IK_USER_AWS_ACCESS_SECRET" ["AWS_REGION=us-east-1"]'
    echo "make sure you set IK_USER_* properties in host environment"
}

function extract_arguments () {
    for arg in $@
    do
        key=$(echo $arg | cut -d '=' -f 1)
        value=$(echo $arg | cut -d '=' -f 2-)
        export $key="$value"
    done
}

function configure () {
    aws configure set aws_access_key_id $AWS_ACCESS_KEY --profile $AWS_PROFILE
    aws configure set aws_secret_access_key $AWS_ACCESS_SECRET --profile $AWS_PROFILE
    aws configure set region $AWS_REGION --profile $AWS_PROFILE
    aws configure set output yaml --profile $AWS_PROFILE
}

function print () {
    echo "---------------------------------------"
    echo "HOME = $(echo `pwd`)"
    aws configure list
    echo "---------------------------------------"
}

extract_arguments $@

if [ -z "$AWS_PROFILE" ]; then
    echo 'AWS_PROFILE is not passed as an argument! `e.g. AWS_PROFILE=john`'
    usage
    exit 1
fi

if [ -z "$AWS_ACCESS_KEY" ]; then
    echo 'AWS_ACCESS_KEY is not passed as an argument! `e.g. AWS_ACCESS_KEY=key`'
    usage
    exit 1
fi

if [ -z "$AWS_ACCESS_SECRET" ]; then
    echo 'AWS_ACCESS_SECRET is not passed as an argument! `e.g. AWS_ACCESS_SECRET=secret`'
    usage
    exit 1
fi

if [ -z "$AWS_REGION" ]; then
    export AWS_REGION="us-east-1"
    echo "defaulting AWS_REGION to us-east-1"
fi

configure
print
/bin/bash
