# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pickle
from datetime import datetime

from mango.messages.codecs import JSON, GenericProtoMsg

from assume.common.utils import datetime2timestamp, timestamp2datetime


def datetime_json_serializer():
    def __tostring__(dt: datetime):
        return datetime2timestamp(dt)

    def __fromstring__(ts: float | str):
        return timestamp2datetime(ts)

    return datetime, __tostring__, __fromstring__


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
    codec.add_serializer(*datetime_json_serializer())
    codec.add_serializer(*generic_json_serializer())
    return codec
