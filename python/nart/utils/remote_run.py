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

import sys
import io
import os
import signal

interrupted = False


def check_interrupt():
    if interrupted:
        exit(0)


def check_dir(dir_name):
    folder = os.path.exists(dir_name)
    if not folder:
        os.makedirs(dir_name)
        print("Create path success:{}".format(dir_name))


class RemoteShell:
    def __init__(self, hostname, port, username, password):
        try:
            import paramiko

            self.client = paramiko.SSHClient()
            self.client.load_system_host_keys()
            self.client.set_missing_host_key_policy(paramiko.WarningPolicy())
            print("Connect Server: {}:{} user:{}".format(hostname, port, username))
            self.client.connect(hostname, port, username, password)
            self.sftp = self.client.open_sftp()
            print("Connect SFTP Success.")
        except Exception as e:
            print("*** Caught exception: %s: %s" % (e.__class__, e))
            try:
                self.client.close()
            except:
                pass

    def __del__(self):
        try:
            print("Disconnect SSH !")
            self.client.close()
        except:
            pass

    def run_command(self, cmd):
        try:
            print("Current dir:", self.sftp.getcwd(), "Run commond:", cmd)
            stdin, stdout, stderr = self.client.exec_command(cmd, get_pty=True)
            print("Output content:")
            for line in iter(stdout.readline, ""):
                check_interrupt()
                print(line, end="", flush=True)

            stderr_lines = stderr.readlines()
            if len(stderr_lines) != 0:
                print("Error content:")
                for line in stderr_lines:
                    check_interrupt()
                    print(line, end="", flush=True)
            status = stdout.channel.recv_exit_status()
            print("Return Status:", status)
            return status
        except Exception as e:
            print("*** Caught Running exception: %s: %s" % (e.__class__, e))

    def sftp_put(self, local_file, remote_file):
        try:
            check_interrupt()
            print("Upload file {} to {}.".format(local_file, remote_file))
            self.sftp.put(local_file, remote_file)
            print("Upload file {} to {} success.".format(local_file, remote_file))
        except Exception as e:
            print("*** Upload file exception: %s: %s" % (e.__class__, e))

    def sftp_get(self, remote_file, local_file):
        try:
            check_interrupt()
            print("Download file {} to {}.".format(remote_file, local_file))
            self.sftp.get(remote_file, local_file)
            print("Download file {} to {} success.".format(remote_file, local_file))
        except Exception as e:
            print("*** Download file exception: %s: %s" % (e.__class__, e))

    def sftp_mkdir(self, remote_dir):
        try:
            check_interrupt()
            self.sftp.mkdir(remote_dir)
            print("Create remote dir:{} success.".format(remote_dir))
        except Exception as e:
            print("*** mkdir exception: %s: %s" % (e.__class__, e))

    def sftp_chdir(self, remote_dir):
        try:
            check_interrupt()
            self.sftp.chdir(path=remote_dir)
            print("Change the current directory: {} ".format(remote_dir))
        except Exception as e:
            print("*** chdir exception: %s: %s" % (e.__class__, e))

    def sftp_rmdir(self, remote_dir):
        try:
            check_interrupt()
            self.run_command("cd /; rm -rf {}".format(remote_dir))
            print("Remove directory: {} ".format(remote_dir))
        except Exception as e:
            print("*** rmdir exception: %s: %s" % (e.__class__, e))
