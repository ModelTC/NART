# Copyright 2022 SenseTime Group Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket
import datetime
import getpass
import platform
import os
import numpy as np
from numpy.distutils import cpuinfo


def format_message(custom_info=None):
    host = socket.gethostname()
    user = getpass.getuser()
    cpu = cpuinfo.cpu.info[0]["model name"]
    os = platform.platform()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
    except:
        ip = socket.gethostbyname(host)
    timestamp = datetime.datetime.now()
    message = "Host: {}, User: {}, CPU: {}, OS: {}, IP: {}, Timestamp: {}".format(
        host, user, cpu, os, ip, timestamp
    )
    if custom_info is not None:
        for key, value in custom_info.items():
            message += ", {}: {}".format(key, value)
    return message


def extract_message(message, keyword):
    try:
        tokens = message.split(", ")
        for token in tokens:
            key, val = token.split(": ")
            if keyword.lower() == key.lower():
                return val
    except:
        return "Unidentified {}".format(keyword)


def save(message):
    dirname = os.path.dirname(os.path.realpath(__file__))
    dirname = os.path.join(dirname, "log")
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    host = extract_message(message, "Host")
    user = extract_message(message, "User")
    filename = "{}_{}.csv".format(host, user)
    filepath = os.path.join(dirname, filename)
    if os.path.exists(filepath):
        data = np.loadtxt(filepath, delimiter="\n", dtype=str, ndmin=1)
        data = np.concatenate((data, [message]))
    else:
        data = np.array([message])
    np.savetxt(filepath, data, delimiter="\n", fmt="%s")


def send(method, host="10.10.40.93", port=16210):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(3.0)
        message = format_message(method)
        try:
            sock.connect((host, port))
            sock.sendall(message.encode())
        except Exception:
            save(message)
            return
        dirname = os.path.dirname(os.path.realpath(__file__))
        dirname = os.path.join(dirname, "log")
        # decache
        if os.path.exists(dirname):
            csv_names = [
                filename
                for filename in os.listdir(dirname)
                if filename.endswith(".csv")
            ]
            for filename in csv_names:
                filepath = os.path.join(dirname, filename)
                data = np.loadtxt(filepath, delimiter="\n", dtype=str, ndmin=1)
                try:
                    while data.shape[0] > 0:
                        data, message = data[:-1], data[-1]
                        sock.sendall(message.encode())
                    if data.shape[0] == 0:
                        os.remove(filepath)
                except Exception:
                    os.remove(filepath)
                    save(message)
                    for message in data:
                        save(message)
                    return
            if len(os.listdir(dirname)) == 0:
                os.rmdir(dirname)
            sock.close()
