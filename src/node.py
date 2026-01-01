import json
import queue
import socket
import sys
import threading
from queue import Queue
import select
import time

import psutil
import tensorflow as tf

from node_state import NodeState, socket_recv, socket_send
from node_state import StateEnum

import zfpy
import lz4.frame

# 6000 is data, 6001 is model config(weights + architecture), 6003 is results
DATA_PORT = 6000 # receive input data
CONFIG_PORT = 6001 # receive model config(weights + architecture)
RESULT_PORT = 6003 # send results

class Node:
    def __init__(self) -> None:
        # initial a event object to signal when weights are ready
        self.weights_ready_event = threading.Event()

    def _model_socket(self, node_state: NodeState, weights_ready_event: threading.Event):
        model_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        model_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        model_server.bind(("0.0.0.0", 6001))
        print("Model socket running")
        model_server.listen(1)
        model_cli = model_server.accept()[0]
        model_cli.setblocking(0)
        model_json = socket_recv(model_cli, node_state.chunk_size)
        next_node = socket_recv(model_cli, chunk_size=1)

        part = tf.keras.models.model_from_json(model_json)

        # wait until weights are ready
        weights_ready_event.wait()

        part.set_weights(node_state.weights)
        id = self.get_local_ip()
        md = part
        node_state.model = md
        tf.keras.utils.plot_model(md, f"model_{id}.png")
        node_state.next_node = next_node.decode()
        select.select([], [model_cli], [])
        model_cli.send(b'\x06')
        model_server.close()

    def _weights_socket(self, node_state, weights_ready_event: threading.Event):
        chunk_size = node_state.chunk_size
        weights_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        weights_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        weights_server.bind(("0.0.0.0", 6002))
        print("Weights socket running")
        weights_server.listen(1)
        weights_cli = weights_server.accept()[0]
        weights_cli.setblocking(0)

                # Validate received data
                if not model_json or not worker_index_bytes:
                    print("Incomplete config received. Restarting listen.")
                    client_sock.close()
                    continue

                worker_index = int(worker_index_bytes.decode())

                # 3. receive weights
                weights_data = self._recv_weights(client_sock, node_state.chunk_size)

                # 4. construct model partition
                part = tf.keras.models.model_from_json(model_json)
                part.set_weights(weights_data)

                print(f"Model Partition {worker_index} loaded. Enqueuing job...")
                self.job_queue.put((worker_index, part))
                node_state._state_enum = StateEnum.BUSY

                # Optional: save visualization
                # tf.keras.utils.plot_model(part, f"model_{id}_{worker_index}.png")

                # 5. send ACK
                client_sock.send(b'\x06')  # ACK
                time.sleep(0.05) # ensure ACK is sent before closing
                client_sock.close()
                print("Model & Weights received and loaded successfully. Ready for inference.")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Config socket error: {e}. Restarting listen.")
            finally:
                config_server.close()

    """ Receive weights over socket """
    def _recv_weights(self, sock: socket.socket, chunk_size: int):
        size_left = 8
        byts = bytearray()
        while size_left > 0:
            try:
                recv = sock.recv(min(size_left, 8))
                size_left -= len(recv)
                byts.extend(recv)
            except socket.error as e:
                if e.errno != socket.EAGAIN:
                    raise e
                select.select([sock], [], [])
        array_len = int.from_bytes(byts, 'big')

        weights = []
        for i in range(array_len):
            recv = bytes(socket_recv(sock, chunk_size))
            weights.append(self._decomp(recv))
        return weights

    """ Helper Functions for Compression/Decompression """
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        return zfpy.decompress_numpy(lz4.frame.decompress(byts))

    """
    Runs the data receiving server. This server accepts incoming connections
    (typically from a Dispatcher), reads a 4-byte index header, receives the
    data payload using socket_recv, decompresses it, and puts the (index, data)
    pair into the processing queue (to_send).

    :param self: The containing object instance (e.g., a distributed node).
    :param node_state: Object containing configuration like chunk_size.
    :param to_send: The queue to place the received (target_index, input_data) for processing.
    """
    def _data_server(self, node_state: 'NodeState', to_send: Queue):
        chunk_size = node_state.chunk_size
        # 1. Socket Initialization and Setup
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allows the socket to be immediately re-bound after closure.
        data_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_server.bind(("0.0.0.0", 6000))
        data_server.listen(1)
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        try:
            # Bind the socket to all available interfaces on the predefined port.
            data_server.bind(("0.0.0.0", DATA_PORT))
            data_server.listen(5)  # Set connection queue limit
            print("Data server running...")
        except Exception as e:
            print(f"Bind error: {e}")
            return

        # 2. Main Server Loop
        while True:
            data = bytes(socket_recv(data_cli, chunk_size))
            inpt = self._decomp(data)
            to_send.put(inpt)

    def _data_client(self, node_state: NodeState, to_send: Queue):
        while node_state.next_node == "":
            time.sleep(5)# Wait until next_node is set by model socket
        chunk_size = node_state.chunk_size
        model = node_state.model
        next_node_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # localhost for testing, in real deployment this would be the next node's IP
        next_node_client.connect((node_state.next_node, 6003))
        next_node_client.setblocking(0)

        while True:
            print("Data client waiting for data")
            inpt = to_send.get()
            print("Data cclient received data")
            output = model.predict(inpt)
            out = self._comp(output)
            socket_send(out, next_node_client, chunk_size)

    def run(self):
        ns = NodeState(chunk_size=512 * 1000)
        # pass the event to the threads that need it
        m = threading.Thread(target=self._model_socket, args=(ns, self.weights_ready_event))
        w = threading.Thread(target=self._weights_socket, args=(ns, self.weights_ready_event))
        to_send = queue.Queue(1000)

        # Start data server and data client threads 6000
        dserv = threading.Thread(target=self._data_server, args=(ns, to_send), daemon=True)
        dcli = threading.Thread(target=self._data_client, args=(ns, to_send), daemon=True)

        config_thread.start()
        dserv.start()
        dcli.start()


    def get_local_ip(self):
        # get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # try to connect to an external IP address
            s.connect(('1.1.1.1', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'  # fallback to localhost
        finally:
            s.close()
        return IP

node = Node()
node.run()

