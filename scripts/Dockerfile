FROM amazon/aws-cli:2.0.6
RUN yum install -y util-linux

RUN mkdir /root/.ssh
RUN yum install -y openssh-clients

COPY ./aws/aws_config /root/.aws/config
RUN ls /root/.aws/config

RUN mkdir /aws/scripts
ADD ./aws /aws/scripts
RUN chmod -R 755 /aws/scripts
RUN ls -ltr /aws/scripts

# RUN echo 'export AWS_PROFILE=ik_user' >> /root/.bashrc
# RUN cat /root/.bashrc
# RUN source /root/.bashrc

ENTRYPOINT ["/aws/scripts/validate_and_start_bash.sh"]