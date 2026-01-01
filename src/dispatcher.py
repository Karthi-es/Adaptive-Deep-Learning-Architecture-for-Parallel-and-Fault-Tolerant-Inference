import select
import socket
import threading
from typing import List
import queue

from .dag_util import *
from .node_state import socket_recv, socket_send

import lz4.frame
import zfpy
import time

# 6000 is data, 6001 is model config(weights + architecture), 6003 is results
DATA_PORT = 6000 # send input data
CONFIG_PORT = 6001 # send model config(weights + architecture)
RESULT_PORT = 6003 # receive results


class DEFER:
    def __init__(self, computeNodes) -> None:
        self.computeNodes = computeNodes
        self.dispatchIP = self.get_local_ip()
        self.chunk_size = 512 * 1000

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
    
    def _partition(self, model: tf.keras.Model, layer_parts: List[str]) -> List[tf.keras.Model]:
        models = []
        for p in range(len(layer_parts) + 1):
            if p == 0:
                start = model.input._keras_history[0].name
            else:
                start = layer_parts[p-1]
            if p == len(layer_parts):
                print(model.output)
                end = model.output._keras_history[0].name
            else:
                end = layer_parts[p]
            part = construct_model(model, start, end, part_name=f"part{p+1}")
            models.append(part)
        return models

    def _dispatchModels(self, models: list, nodeIPs: List[str]) -> None:
        for i in range(len(models)):
            weights_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            weights_sock.setblocking(0)
            weights_sock.settimeout(10)
            model_json = models[i].to_json()
            weights_sock.connect((nodeIPs[i], 6002))
            if i != len(models) - 1:
                nextNode = nodeIPs[i + 1]
            else:
                start = layer_parts[p-1]
            if p == len(layer_parts):
                print(model.output)
                end = model.output._keras_history[0].name
            else:
                end = layer_parts[p]
            part = construct_model(model, start, end, part_name=f"part{p+1}")
            models.append(part)
        return models

    """ Send model weights to a worker """
    def _send_weights(self, weights: List, sock: socket.socket, chunk_size: int):
        size = len(weights)
        size_bytes = size.to_bytes(8, 'big')
        while len(size_bytes) > 0:
            try:
                sent = sock.send(size_bytes)
                size_bytes = size_bytes[sent:]
            except socket.error as e:
                    if e.errno != socket.EAGAIN:
                        raise e
                    select.select([], [sock], [])
        for w_arr in weights:
                as_bytes = self._comp(w_arr)
                socket_send(as_bytes, sock, chunk_size)

    """ Helper functions for compression and decompression"""
    def _comp(self, arr):
        return lz4.frame.compress(zfpy.compress_numpy(arr))
    def _decomp(self, byts):
        # decompress lz4
        zfp_data = lz4.frame.decompress(byts)
        # decompress zfp
        return zfpy.decompress_numpy(zfp_data)
    def _startDistEdgeInference(self, input: queue.Queue):
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_sock.connect((self.computeNodes[0], 6000))
        data_sock.setblocking(0)
        
        while True:
            model_input = input.get()
            out = self._comp(model_input)
            socket_send(out, data_sock, self.chunk_size)

    def _result_server(self, output: queue.Queue):
        data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        data_server.bind(("0.0.0.0", 6003))
        data_server.listen(1) 
        data_cli = data_server.accept()[0]
        data_cli.setblocking(0)

        while True:
            data = bytes(socket_recv(data_cli, self.chunk_size))
            pred = self._decomp(data)
            output.put(pred)

                        # close socket
                        if current_worker_ip:
                            completed_task_key = (current_worker_ip, worker_index)
                            with self.inflight_lock:
                                if completed_task_key in self.inflight_tasks:
                                    # print(f"DEBUG: Task {completed_task_key} finished. Removing from watchdog.")
                                    del self.inflight_tasks[completed_task_key]
                                else:
                                    print(f"WARNING: Could not find task {completed_task_key} to delete! Keys in registry: {list(self.inflight_tasks.keys())}")
                                    pass

                        # 3. Check logic
                        next_worker_index = worker_index + 1

                        num_partitions = len(self.models_to_dispatch) if self.models_to_dispatch else 0

                        if next_worker_index <= num_partitions:
                            # forward to next partition
                            t = threading.Thread(
                                target=self._wait_and_forward,
                                args=(next_worker_index, data),
                                daemon=True
                            )
                            t.start()
                        else:
                            # final partition, put result into output queue
                            print(f"Received final result from Partition {worker_index}. Putting into output queue.")
                            pred = self._decomp(data)
                            output_stream.put(pred)
                            print("Image processing complete. Releasing concurrency slot.")
                            self.concurrency_sem.release()

                    except socket.error as e:
                        s.close()
                        if s in worker_cli:
                            del worker_cli[s]
                    except Exception as e:
                        print(f"Unexpected error in server loop: {e}")
                        s.close()
                        if s in worker_cli:
                            del worker_cli[s]

    """ Update localhost mapping for incoming connections """
    def _update_localhost(self, conn, raw_ip, worker_cli):
        final_ip = raw_ip
        with self.worker_lock:
            # check if raw_ip is in computeNodes
            if raw_ip not in self.computeNodes and '127.0.0.1' in self.computeNodes:
                # map to localhost
                # print(f"DEBUG: Mapping incoming connection from {raw_ip} to 127.0.0.1")
                final_ip = '127.0.0.1'
        # update mapping
        worker_cli[conn] = final_ip

    """ Wait for an available worker and forward data to it """
    def _wait_and_forward(self, next_worker_index, data):
        # 1. dynamically acquire and configure the next worker
        target_ip = self._acquire_and_configure_worker(partition_index=next_worker_index)

        if not target_ip:
            print(f"Stopped forwarding P{next_worker_index} due to shutdown/error.")
            # release semaphore since we cannot proceed
            self.concurrency_sem.release()
            return

        # 2. register in-flight task
        task_key = (target_ip, next_worker_index)

        with self.inflight_lock:
            self.inflight_tasks[task_key] = {
                'partition': next_worker_index, # partition index
                'data': data, # raw data
                'start_time': time.time() # timestamp
            }

        # 3. send data asynchronously
        try:
            print(f"Async Forwarding data to P{next_worker_index} ({target_ip})...")
            self._forward_data_to_worker(target_ip, data, next_worker_index)
        except Exception as e:
            print(f"Error in async forwarding to {target_ip}: {e}")

    """ Forward data to next worker """
    def _forward_data_to_worker(self, worker_ip, data, target_index):
        forward_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            forward_sock.connect((worker_ip, DATA_PORT))

            # 1. send target partition index as 4-byte big-endian integer
            forward_sock.sendall(int(target_index).to_bytes(4, 'big'))

            # 2. send data
            socket_send(data, forward_sock, self.chunk_size)

            # 3. shutdown write
            forward_sock.shutdown(socket.SHUT_WR)
        except Exception as e:
            print(f"Error forwarding data to {worker_ip}: {e}")
        finally:
            forward_sock.close()

    """ Send full model configuration to a worker """
    def _send_full_configuration(self, worker_ip, model_part, worker_index_val):
        model_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 1. timeout for connect
        model_sock.settimeout(5)

        try:
            model_sock.connect((worker_ip, CONFIG_PORT))

            # 2. send data in blocking mode
            model_sock.setblocking(1)

            # send model JSON
            model_json = model_part.to_json()
            socket_send(model_json.encode(), model_sock, self.chunk_size)

            # send worker index
            index_bytes = str(worker_index_val).encode()
            socket_send(index_bytes, model_sock, chunk_size=1)

            # send weights
            self._send_weights(model_part.get_weights(), model_sock, self.chunk_size)

            # 3. set timeout for ACK
            model_sock.settimeout(10)

            # 4. wait for ACK
            # set 60 seconds timeout for select
            ready = select.select([model_sock], [], [], 60)
            if ready[0]:
                try:
                    ack = model_sock.recv(1)
                    if ack == b'\x06':
                        print(f"Worker {worker_ip} acknowledged configuration.")
                    else:
                        # if not ACK, raise error
                        print(f"Worker {worker_ip} sent unexpected ACK: {ack}")
                except socket.timeout:
                    raise socket.timeout("Timed out waiting for Worker ACK (Blocking Read)")
        except Exception as e:
            raise e
        finally:
            model_sock.close()

    """
    Main entry point for DEFER dispatcher
    :param model: The full Keras model to be partitioned and dispatched.
    :param partition_layers: List of layer names where the model should be partitioned.
    :param input_stream: Queue for incoming input data.
    :param output_stream: Queue for outgoing results.
    """
    def run_defer(self, model: tf.keras.Model, partition_layers, input_stream: queue.Queue, output_stream: queue.Queue):

        # 1. start worker monitor thread
        monitor_thread = threading.Thread(target=self._worker_monitor, daemon=True)
        monitor_thread.start()

        # 2. partition model
        self.models_to_dispatch = self._partition(model, partition_layers)

        # 3. wait for initial workers (wait up to 5 seconds)
        max_wait = 5
        for i in range(max_wait):
            initial_workers = self._get_available_workers()
            if initial_workers:
                with self.worker_lock:
                    self.computeNodes = initial_workers
                break
            time.sleep(1)
        else:
            print(f"Error: No workers registered after {max_wait} seconds.")
            # set shutdown event to terminate other threads
            self._shutdown_event.set()
            return

        # 4. start intermediate result server
        intermediate_thread = threading.Thread(target=self._intermediate_result_server, args=(output_stream,),
                                               daemon=True)
        intermediate_thread.start()

        # 5. start task watchdog
        wd_thread = threading.Thread(target=self._task_watchdog, daemon=True)
        wd_thread.start()

        # 6. start distributed edge inference
        start_thread = threading.Thread(target=self._startDistEdgeInference, args=(input_stream,), daemon=True)
        start_thread.start()

        # 7. keep main thread alive
        try:
            while not self._shutdown_event.is_set():
                time.sleep(2)
        except KeyboardInterrupt:
            print("Dispatcher shutting down...")
        finally:
            self._shutdown_event.set()