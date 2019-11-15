import os
import logging
import socket

DEBUG =0

SERVER_PORT = 5911
PORT = 5922

WARMUP=True


PREFIX=''
if WARMUP:
    PREFIX=PREFIX+'gwarmup'

LOGGING_ASSUMPTION=False
LOGGING_GRADIENTS=False

EXP='-convergence'
PREFIX=PREFIX+EXP
ADAPTIVE_MERGE=False
ADAPTIVE_SPARSE=False
if ADAPTIVE_MERGE:
    PREFIX=PREFIX+'-ada'

TENSORBOARD=False
USE_FP16=False

MAX_EPOCHS=10

hostname=socket.gethostname() 
logger=logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

