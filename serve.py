#!/usr/bin/python

import http.server
import socketserver
import os

PORT = 8080

os.chdir('output')
Handler = http.server.S