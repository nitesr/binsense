#!/bin/bash

CLEAN_UP=0
CREATE=0
PROFILE=$AWS_PROFILE


PROFILE_ARG_PATTERN="--profile=[^=]+"
EC2_INSTANCE_TYPE_ARG_PATTERN="--instance=[^=]+"
for arg in "$@"
do
    if [ "$arg" = "--clean" ]; then
        CLEAN_UP=1
    fi

    if [ "$arg" = "--create" ]; then
        CREATE=1
    fi

    if [[ "$arg" =~ $PROFILE_ARG_PATTERN ]]; then
        PROFILE=$(echo $arg | cut -d '=' -f 2)
    fi

    if [[ "$arg" =~ $EC2_INSTANCE_TYPE_ARG_PATTERN ]]; then
        EC2_INSTANCE_TYPE_ARG_VALUE=$(echo $arg | cut -d '=' -f 2)
    fi
done


SG_NAME=$PROFILE"_SecurityGroup"
KEYPAIR_NAME=$PROFILE"_KeyPair"
DL_EC2_NAME=$PROFILE"_DL_EC2"

EC2_AMI_ID="ami-0da2ab58cace8997d"
EC2_INSTANCE_TYPE=$( [[ ! "$EC2_INSTANCE_TYPE_ARG_VALUE" == "" ]] && echo "$EC2_INSTANCE_TYPE_ARG_VALUE" || echo "g4dn.12xlarge")

function remove_null () {
    [ "$1" = "null" ] && echo "" || echo "$@"
}

function find () {
    VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --profile $PROFILE 2>/dev/null | tr -d '"')
    VPC_ID=$(remove_null $VPC_ID)

    SG_ID=$(aws ec2 describe-security-groups --group-name $SG_NAME --query 'SecurityGroups[0].GroupId' --profile $PROFILE  2>/dev/null | tr -d '"')
    SG_ID=$(remove_null $SG_ID)

    PUBLIC_DNS_NAME=$(aws ec2 describe-instances --filter "Name=tag:Name,Values=$DL_EC2_NAME" --output text --query 'Reservations[*].Instances[?State.Code == `16`].PublicDnsName[]' --profile $PROFILE  2>/dev/null | tr -d '"')
    PUBLIC_DNS_NAME=$(remove_null $PUBLIC_DNS_NAME)

    EC2_INSTANCE_IDS=$(aws ec2 describe-instances --filter "Name=tag:Name,Values=$DL_EC2_NAME" --output text --query 'Reservations[*].Instances[?State.Code == `16`].InstanceId[]' --profile $PROFILE  | tr -d '"')
    EC2_INSTANCE_IDS=$(remove_null $EC2_INSTANCE_IDS)

    echo "---------------------------------------"
    echo "KEYPAIR_NAME = $KEYPAIR_NAME"
    echo "VPC_ID = $VPC_ID"
    echo "SG_ID = $SG_ID"
    echo "PUBLIC_DNS_NAME = $PUBLIC_DNS_NAME"
    echo "EC2_INSTANCE_IDS = $EC2_INSTANCE_IDS"
    echo "---------------------------------------"
}

function is_vpc_exist () {
    if [ -z "$VPC_ID" ]; then
        return 1
    else
        return 0
    fi
}

function is_key_pair_exist () {
    aws ec2 describe-key-pairs --key-name $KEYPAIR_NAME --profile $PROFILE  2>/dev/null
    return $?
}

function is_sg_exist () {
    aws ec2 describe-security-groups --group-name $SG_NAME --profile $PROFILE  2>/dev/null
    return $?
}

function is_ec2_running () {
    aws ec2 describe-instances --filter "Name=tag:Name,Values=$DL_EC2_NAME" --profile $PROFILE  1>/dev/null 2>/dev/null
    if [ $? -eq 0 ]; then
        local not_terminated_ec2_instance_ids=$(aws ec2 describe-instances --filter "Name=tag:Name,Values=$DL_EC2_NAME" --output text --query 'Reservations[*].Instances[?State.Code == `16`].InstanceId[]' --profile $PROFILE 2>/dev/null | tr -d '"')
        not_terminated_ec2_instance_ids=$(remove_null $not_terminated_ec2_instance_ids)

        if [ ! -z "$not_terminated_ec2_instance_ids" ]; then
            echo "$DL_EC2_NAME EC2 instance are not terminated! id=$not_terminated_ec2_instance_ids"
            return 0
        fi
        return 1
    fi
    return $?
}

function validate () {
    is_key_pair_exist
    if [ $? -eq 0 ]; then
        echo "$KEYPAIR_NAME key-pair exists"
        exit 1
    fi

    is_sg_exist
    if [ $? -eq 0 ]; then
        echo "$SG_NAME security group exists"
        exit 1
    fi

    is_ec2_running
    if [ $? -eq 0 ]; then
        echo "$DL_EC2_NAME EC2 instance is running!"
        exit 1
    fi

    is_vpc_exist
    if [ $? -gt 0 ]; then
        echo "There is no default VPC set!"
        exit 1
    fi
}

function cleanup () {
    if [ ! -z "$EC2_INSTANCE_IDS" ]; then
        aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_IDS --profile $PROFILE 1>/dev/null 2>/dev/null
        [ $? -eq 0 ] && echo "$DL_EC2_NAME EC2 instance terminated, id= $EC2_INSTANCE_IDS"
    fi

    if [ ! -z "$SG_ID" ]; then
        aws ec2 delete-security-group --group-name $SG_NAME --profile $PROFILE 1>/dev/null 2>/dev/null
        [ $? -eq 0 ] && echo "$SG_NAME SecurityGroup deleted, id=$SG_ID" || echo "ERR: $SG_NAME SecurityGroup is not deleted, id=$SG_ID"
    fi

    aws ec2 delete-key-pair --key-name $KEYPAIR_NAME --profile $PROFILE 1>/dev/null 2>/dev/null
    [ $? -eq 0 ] && echo "$KEYPAIR_NAME key-pair deleted"
}



function create_key_pair () {
    aws ec2 create-key-pair --key-name $KEYPAIR_NAME --query 'KeyMaterial' --profile $PROFILE | tr -d '"' | sed 's/\\n/\n/g' > ~/.ssh/$KEYPAIR_NAME.pem

    if [ $? -eq 0 ]; then
        echo "$KEYPAIR_NAME key-pair created and private key is stored at ~/.ssh/$KEYPAIR_NAME.pem"
        chmod 400 ~/.ssh/$KEYPAIR_NAME.pem
    fi
}

function create_sg () {
    aws ec2 create-security-group --group-name $SG_NAME --description "default security group" --vpc-id $VPC_ID --output yaml --profile $PROFILE 1> `echo $SG_NAME`_create.log
    if [ $? -eq 0 ]; then
        SG_ID=$(aws ec2 describe-security-groups --group-name $SG_NAME --query 'SecurityGroups[0].GroupId' --profile $PROFILE | tr -d '"')
        echo "$SG_NAME security group is created, id=$SG_ID"

        aws ec2 authorize-security-group-ingress --group-name $SG_NAME --ip-permissions "[ {  \"IpProtocol\":  \"tcp\",  \"FromPort\": 22,  \"ToPort\": 22,  \"IpRanges\": [ {  \"CidrIp\":  \"0.0.0.0/0\" } ] }, {  \"IpProtocol\":  \"tcp\",  \"FromPort\": 6600,  \"ToPort\": 6600,  \"IpRanges\": [ {  \"CidrIp\":  \"0.0.0.0/0\" } ] } ]" --output yaml --profile $PROFILE
        [ $? -eq 0 ] && echo "ingres set to ssh from any port for $SG_NAME security group"
    fi
}

function create_ec2 () {
    aws ec2 run-instances --image-id $EC2_AMI_ID --count 1 --instance-type $EC2_INSTANCE_TYPE --key-name $KEYPAIR_NAME --ebs-optimized --block-device-mappings "[ {  \"DeviceName\":  \"/dev/sda1\",  \"Ebs\": {  \"Encrypted\": false,  \"DeleteOnTermination\": true,  \"Iops\": 3000,  \"SnapshotId\":  \"snap-0548faad3e87fff59\",  \"VolumeSize\": 100,  \"VolumeType\":  \"gp3\" } } ]" --network-interfaces "[ { \"AssociatePublicIpAddress\": true, \"DeviceIndex\": 0, \"Groups\": [ \"$SG_ID\" ] }]" --metadata-options "{ \"HttpTokens\": \"required\", \"HttpEndpoint\": \"enabled\", \"HttpPutResponseHopLimit\": 2}" --tag-specifications "[ { \"ResourceType\": \"instance\", \"Tags\": [ { \"Key\": \"Name\", \"Value\": \"$DL_EC2_NAME\" } ] }]" --output yaml --profile $PROFILE 1> `echo $DL_EC2_NAME`_create.log

    if [ $? -eq 0 ]; then
        PUBLIC_DNS_NAME=$(aws ec2 describe-instances --filter "Name=tag:Name,Values=$DL_EC2_NAME" --query 'Reservations[-1].Instances[0].PublicDnsName' --profile $PROFILE 2>/dev/null | tr -d '"')

        EC2_INSTANCE_ID=$(aws ec2 describe-instances --filter "Name=tag:Name,Values=$DL_EC2_NAME" --query 'Reservations[-1].Instances[0].InstanceId' --profile $PROFILE 2>/dev/null | tr -d '"')

        echo "$DL_EC2_NAME EC2 instance is launched, PUBLIC_DNS_NAME=$PUBLIC_DNS_NAME, id=$EC2_INSTANCE_ID. It will take a while for instance to start. You can execute this script without any options to check back."
    fi
}

find

if [ $CLEAN_UP -ne 0 ]; then
    echo "setup cleans $KEYPAIR_NAME, $SG_NAME, $DL_EC2_NAME resources."
    cleanup
    find
    cleanup
fi

if [ $CREATE -ne 0 ]; then
    validate

    echo "creating $KEYPAIR_NAME, $SG_NAME, $DL_EC2_NAME($EC2_INSTANCE_TYPE) ..."
    create_key_pair
    create_sg
    create_ec2

    find
    if [ -z "$PUBLIC_DNS_NAME" ]; then
        echo "ec2 instance not available yet, waiting..."
        sleep .5
        find
    fi
fi