# Basic MNIST Example with RedisClient

Tested Python 3.7.7 and 3.8.2 with PyTorch 1.4.0

## Redis server installation

### install package
```
$ sudo apt update
$ sudo apt install redis-server
```
### overcommit memory
edit `/etc/sysctl.conf` to add:
```
vm.overcommit_memory = 1
```
and reboot or run the command
```
$ sudo sysctl vm.overcommit_memory=1
```
for this to take effect.

### transparent hugh pages (THP)
If THP support enabled in your kernal, this will create latency and memory usage issues with Redis. Run the command
```
# echo never > /sys/kernel/mm/transparent_hugepage/enabled
```
or add it to your `/etc/rc.local` in order to retain the settings after a reboot.

### disable save to disk

disable the aof
```
# redis-cli config set appendonly no
```
disable the rdb
```
# redis-cli config set save ""
(default was "900 1 300 10 60 10000")
```
If you want to make these changes effective after restarting redis, using
```
# redis-cli config rewrite
```

## Run MNIST example

### Preparing dataset
```bash
$ pip install -r requirements.txt
$ python dataset.py
```

### Training
```bash
$ python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

## Comments

1. There exists [RedisLab's official Redis module for PyTorch](https://github.com/RedisAI/RedisAI), but it only supports tensor type to store.
   Using this project, you can store any structured data associated to a key such as a list of tensors or a list of tuples of tensors mixed with strings etc.
2. In my experiment, `tensor.numpy()` has smaller memory footprint than `numpy ndarray`.
3. If `num_workers=0` in DataLoader, it is inevitably much slower than direct-access of in-memory data. Use multipe `num_workers` for the performance.

## Benchmarks

| Env                       | num_workers | elapsed time (15 epochs) |
| ------------------------- | ----------- | ------------------------ |
| torchvision MNIST dataset | 0           | 99.9 secs                |
| RedisClient               | 4           | 350.4 secs               |
| RedisClient               | 8           | 184.0 secs               |
| RedisClient               | 16          | 116.6 secs               |
