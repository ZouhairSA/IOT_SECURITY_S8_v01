from pymongo import MongoClient

class MongoDBConnection:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="pack_iot"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection(self, collection_name):
        return self.db[collection_name]

    def insert_data(self, collection_name, data):
        collection = self.get_collection(collection_name)
        collection.insert_one(data)

    def close(self):
        self.client.close()