import pickle

from mango.messages.codecs import JSON, GenericProtoMsg


def generic_json_serializer():
    def __tostring__(generic_obj):
        return pickle.dumps(generic_obj).hex()

    def __fromstring__(data):
        return pickle.loads(bytes.fromhex(data))

    return object, __tostring__, __fromstring__


def generic_pb_serializer():
    def __tostring__(generic_obj):
        msg = GenericProtoMsg()
        msg.content = pickle.dumps(generic_obj)
        return msg

    def __fromstring__(data):
        msg = GenericProtoMsg()
        msg.ParseFromString(data)

        return pickle.loads(msg.content)

    return object, __tostring__, __fromstring__


def mango_codec_factory():
    codec = JSON()
    codec.add_serializer(*generic_json_serializer())
    return codec
