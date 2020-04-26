import io
import pickle

import redis
import torch
from tqdm.auto import tqdm


PICKLE_VERSION = 5
CXN = redis.ConnectionPool(host='localhost', port=6379, db=0)


class RedisListObject:

    def __init__(self, name):
        self.name = name

    def __len__(self):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            return rdb.llen(self.name)

    def __setitem__(self, index, value):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            if index >= rdb.llen(self.name):
                raise IndexError
            with io.BytesIO() as buf:
                torch.save(value, buf, pickle_protocol=PICKLE_VERSION)
                rdb.lset(self.name, index, buf.getbuffer())

    def __getitem__(self, index):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            if not rdb.exists(self.name):
                raise redis.DataError(f'Dataset named {self.name} does not exist')
            if index >= rdb.llen(self.name):
                raise IndexError
            with io.BytesIO(rdb.lindex(self.name, index)) as buf:
                return torch.load(buf)

    def append(self, value):
        with io.BytesIO() as buf:
            torch.save(value, buf, pickle_protocol=PICKLE_VERSION)
            #print(len(buf.getvalue()))
            with redis.StrictRedis(connection_pool=CXN) as rdb:
                func = rdb.rpush if rdb.exists(self.name) else rdb.lpush
                func(self.name, buf.getbuffer())

    def delete(self):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            if rdb.exists(self.name):
                rdb.delete(self.name)
            else:
                raise redis.DataError(f'Dataset named {self.name} does not exist')


class RedisClient:

    def get(self, key):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            if rdb.exists(key):
                return RedisListObject(key)
            else:
                raise redis.DataError(f'Dataset named {self.name} does not exist')

    def set_data_list(self, key, values):
        try:
            obj = self.get(key)
            obj.delete()
        except:
            obj = RedisListObject(key)

        for item in tqdm(values, desc=f"storing {key}", dynamic_ncols=True):
            obj.append(item)

    def keys(self):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            return rdb.keys()

    def stats(self):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            return rdb.memory_stats()

    def check_lens(self, nums):
        try:
            for k, v in nums.items():
                obj = self.get(k)
                if v != 0 and len(obj):
                    return False
        except:
            return False

    def flushdb(self):
        with redis.StrictRedis(connection_pool=CXN) as rdb:
            rdb.flushdb()


if __name__ == "__main__":
    c = RedisClient()
    print(c.stats())

    data_list = [tuple(torch.rand(10, 10) for _ in range(10)) for _ in range(10)]
    c.set_data_list("test", data_list)
    print(c.test[0], c.test[1])

    c.delete_all()
    print(c.stats())
