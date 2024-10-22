# Implement (code) a Key value store with transactions.

# Write a Fully funcitonal code in 25-30 min in interview with test cases

# Set
# Get
# Delete are methods in Key value store

# for transactions
# Begin
# Commit
# Rollback

from typing import *

class KVStore:
    def __init__(self):
        # Using Stack, mainly for rollback.
        self.stack = [{}]
        pass
    def get(self, key: Any):
        for i in range(len(self.stack) -1, -1, -1):
            if key in self.stack[i]:
                return self.stack[i][key]
            
    def set(self, key: Any, value: Any):
        self.stack[-1][key] = value
    def delete(self, key: Any, value: Any):
        pass
    def begin(self):
        self.stack.append({})
    def commit(self):
        last_dict = self.stack.pop()
        
        for k, v in last_dict.items():
            self.stack[-1][k] = v
        
    
def test_KVStore():
    kv = KVStore()
    kv.set(1, 3)
    
    assert kv.get(1) == 3
    assert kv.get(1) == None
    
def test_KVStore_Xact():
    kv = KVStore()
    kv.set(0, 100)
    kv.set(1, 11)
    kv.begin()
    assert kv.get(1) == 11
    kv.set(1, 22)
    assert kv.get(1) == 22
    kv.begin()
    kv.set(2, 33)

    assert kv.get(0) == 100
    assert kv.get(1) == 22
    assert kv.get(2) == 33
    
    kv.commit()
    
    assert kv.get(0) == 100
    assert kv.get(1) == 22
    assert kv.get(2) == 33
    
    kv.rollback()
    assert kv.get(0) == 100
    assert kv.get(1) == None
    assert kv.get(2) == None
        